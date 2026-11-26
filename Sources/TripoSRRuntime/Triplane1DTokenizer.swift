import Foundation
import MLX
import MLXNN

public class Triplane1DTokenizer: Module {
    public let planeSize: Int
    public let numChannels: Int
    public let embeddings: MLXArray // parameter
    
    public init(planeSize: Int = 32, numChannels: Int = 1024) {
        self.planeSize = planeSize
        self.numChannels = numChannels
        // [3, H, W, C]
        self.embeddings = MLXRandom.normal([3, planeSize, planeSize, numChannels]) * (1.0 / sqrt(Float(numChannels)))
    }
    
    public func callAsFunction(batchSize: Int) -> MLXArray {
        // Create initial tokens for the transformer:
        // [3, H, W, C] -> repeat to [B, 3, H, W, C] -> flatten to [B, 3 * H * W, C]
        // But the transformer wants [B, C, N] in PyTorch, or [B, N, C] in MLX natively.
        // We will output [B, 3 * H * W, C]
        let expanded = MLX.broadcast(embeddings, to: [batchSize, 3, planeSize, planeSize, numChannels])
        return expanded.reshaped([batchSize, 3 * planeSize * planeSize, numChannels])
    }
    
    public func detokenize(_ tokens: MLXArray) -> MLXArray {
        // After transformer: [B, 3 * H * W, C] -> [B, 3, H, W, C]
        let B = tokens.shape[0]
        return tokens.reshaped([B, 3, planeSize, planeSize, numChannels])
    }
}
