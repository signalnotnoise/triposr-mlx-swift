import Foundation
import MLX
import MLXNN

public class TripoSRModel: Module {
    public let imageTokenizer: DINOv2Model
    public let tokenizer: Triplane1DTokenizer
    public let backbone: LRMTransformer1D
    public let postProcessor: TriplaneUpsampleNetwork

    public let decoder: NeRFMLP
    public let renderer: TripoSRRenderer

    public override init() {
        // DINO-ViT-B/16 checkpoint pos embeds are 14×14+CLS (224@patch16).
        // Runtime images are 512×512; pos embeds are bilinearly interpolated in DINOv2Model.
        self.imageTokenizer = DINOv2Model(
            imgSize: 224,
            patchSize: 16,
            inChannels: 3,
            embedDim: 768,
            depth: 12,
            numHeads: 12
        )

        self.tokenizer = Triplane1DTokenizer(planeSize: 32, numChannels: 1024)

        self.backbone = LRMTransformer1D(
            inChannels: 1024,
            numAttentionHeads: 16,
            attentionHeadDim: 64,
            numLayers: 16,
            crossAttentionDim: 768,
            normNumGroups: 32
        )

        self.postProcessor = TriplaneUpsampleNetwork(inChannels: 1024, outChannels: 40)

        self.decoder = NeRFMLP(inChannels: 120, nNeurons: 64, nHiddenLayers: 9)
        self.renderer = TripoSRRenderer(radius: 0.87)
    }

    /// Loads weights from a .safetensors file.
    public func loadWeights(from url: URL) throws {
        let weights = try loadArrays(url: url)
        // Fail hard on shape mismatches / unused keys so silent load bugs cannot pass parity.
        try self.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.shapeMismatch, .noUnusedKeys]
        )
    }

    /// ImageNet mean/std used by DINOSingleImageTokenizer (expects RGB in [0, 1], NHWC).
    public static let imageNetMean: [Float] = [0.485, 0.456, 0.406]
    public static let imageNetStd: [Float] = [0.229, 0.224, 0.225]

    public static func preprocessImage(_ image: MLXArray) -> MLXArray {
        // image: [B, H, W, C] float RGB roughly in [0, 1]
        let mean = MLXArray(imageNetMean).reshaped([1, 1, 1, 3])
        let std = MLXArray(imageNetStd).reshaped([1, 1, 1, 3])
        return (image - mean) / std
    }

    /// Forward pass generating the Triplanes from a 2D image
    /// - Parameter image: RGB float tensor `[B, H, W, C]` in **[0, 1]** (not ImageNet-normalized).
    /// - Parameter alreadyNormalized: if true, skip ImageNet mean/std (for parity dumps that pre-normalized).
    public func forwardTriplanes(image: MLXArray, alreadyNormalized: Bool = false) -> MLXArray {
        // 1. Process Image to tokens (DINO)
        let input = alreadyNormalized ? image : Self.preprocessImage(image)
        let imageTokens = imageTokenizer(input) // [B, SeqLen, 768]

        let batchSize = image.shape[0]

        // 2. Generate learnable 3D latent sequence
        var latentSequence = tokenizer(batchSize: batchSize) // [B, 3 * H * W, 1024]

        // 3. LRM Transformer (Cross-Attention between Latent and Image)
        latentSequence = backbone(latentSequence, encoderHiddenStates: imageTokens)

        // 4. Detokenize back to 3 planes
        let triplanes = tokenizer.detokenize(latentSequence) // [B, 3, H, W, 1024]

        // 5. Upsample Triplanes
        let upsampledTriplanes = postProcessor(triplanes) // [B, 3, H*2, W*2, 40]

        return upsampledTriplanes
    }

    /// Evaluates density and color for a set of 3D positions using the renderer
    public func evaluate(triplanes: MLXArray, positions: MLXArray) -> (density: MLXArray, color: MLXArray) {
        return renderer.queryTriplane(decoder: decoder, positions: positions, triplanes: triplanes)
    }
}
