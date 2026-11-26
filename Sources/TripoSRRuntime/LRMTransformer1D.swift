import Foundation
import MLX
import MLXNN

public class LRMTransformer1D: Module {
    public let norm: GroupNorm
    public let projIn: Linear
    public let blocks: [LRMBasicTransformerBlock]
    public let projOut: Linear
    
    public init(
        inChannels: Int = 1024,
        numAttentionHeads: Int = 16,
        attentionHeadDim: Int = 64,
        numLayers: Int = 16,
        crossAttentionDim: Int = 768,
        normNumGroups: Int = 32
    ) {
        let innerDim = numAttentionHeads * attentionHeadDim
        
        // Match PyTorch GroupNorm grouping (channels-last layout with last-dim features).
        self.norm = GroupNorm(
            groupCount: normNumGroups,
            dimensions: inChannels,
            eps: 1e-6,
            pytorchCompatible: true
        )
        self.projIn = Linear(inChannels, innerDim)
        
        self.blocks = (0..<numLayers).map { _ in
            LRMBasicTransformerBlock(
                dim: innerDim,
                numAttentionHeads: numAttentionHeads,
                attentionHeadDim: attentionHeadDim,
                crossAttentionDim: crossAttentionDim
            )
        }
        
        self.projOut = Linear(innerDim, inChannels)
    }
    
    public func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        // hiddenStates is originally [B, C, N], so it usually needs transposition
        // TripoSR passes it into GroupNorm, which in PyTorch works on [B, C, *]
        // In MLX, GroupNorm works on the last dimension by default [B, N, C]
        // So we assume hiddenStates is [B, SeqLen, inChannels]
        
        let residual = hiddenStates
        
        var x = norm(hiddenStates)
        x = projIn(x)
        
        // Eval after every block: LRM self-attn is 16×[heads,3072,3072]; fusing
        // multiple layers causes GPU timeout / hang on Apple Silicon.
        for (i, block) in blocks.enumerated() {
            x = block(x, encoderHiddenStates: encoderHiddenStates)
            eval(x)
            if i % 4 == 3 {
                print("  LRM block \(i + 1)/\(blocks.count)")
            }
        }
        
        x = projOut(x)
        
        return x + residual
    }
}
