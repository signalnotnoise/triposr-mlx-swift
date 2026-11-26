import Foundation
import MLX
import MLXNN

public class DINOv2Attention: Module {
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float
    
    public let qkv: Linear
    public let proj: Linear
    
    public init(dim: Int, numHeads: Int = 12, qkvBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(headDim).squareRoot()
        
        self.qkv = Linear(dim, dim * 3, bias: qkvBias)
        self.proj = Linear(dim, dim)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let N = x.shape[1]
        
        // [B, N, 3 * dim]
        let qkvOut = qkv(x)
        
        // Split into Q, K, V
        // Reshape to [B, N, 3, numHeads, headDim] -> transpose to [3, B, numHeads, N, headDim]
        let qkvReshaped = qkvOut
            .reshaped([B, N, 3, numHeads, headDim])
            .transposed(axes: [2, 0, 3, 1, 4])
        
        // qkvReshaped is [3, B, numHeads, N, headDim]
        let q = qkvReshaped[0]
        let k = qkvReshaped[1]
        let v = qkvReshaped[2]
        
        // SDPA expects [B, heads, T, D]; scale applied inside (do not pre-scale Q).
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        
        // [B, numHeads, N, headDim] -> [B, N, numHeads, headDim] -> [B, N, dim]
        let xOut = attnOut.transposed(axes: [0, 2, 1, 3]).reshaped([B, N, numHeads * headDim])
        
        return proj(xOut)
    }
}
