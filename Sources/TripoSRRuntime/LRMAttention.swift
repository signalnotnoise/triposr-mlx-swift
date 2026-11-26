import Foundation
import MLX
import MLXNN

public class LRMAttention: Module {
    public let heads: Int
    public let headDim: Int
    public let scale: Float
    
    public let toQ: Linear
    public let toK: Linear
    public let toV: Linear
    public let toOut: Linear
    
    public init(
        queryDim: Int, 
        crossAttentionDim: Int? = nil, 
        heads: Int = 8, 
        dimHead: Int = 64, 
        bias: Bool = false, 
        outBias: Bool = true
    ) {
        self.heads = heads
        self.headDim = dimHead
        let innerDim = dimHead * heads
        self.scale = pow(Float(dimHead), -0.5)
        
        let crossDim = crossAttentionDim ?? queryDim
        
        self.toQ = Linear(queryDim, innerDim, bias: bias)
        self.toK = Linear(crossDim, innerDim, bias: bias)
        self.toV = Linear(crossDim, innerDim, bias: bias)
        
        // to_out in PyTorch has a Linear and Dropout. We just need Linear.
        self.toOut = Linear(innerDim, queryDim, bias: outBias)
    }
    
    public func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        let B = hiddenStates.shape[0]
        
        // If encoderHiddenStates is nil, it's Self-Attention
        let context = encoderHiddenStates ?? hiddenStates
        
        // 1. Projections
        let q = toQ(hiddenStates)
        let k = toK(context)
        let v = toV(context)
        
        let targetLen = q.shape[1]
        let sourceLen = k.shape[1]
        
        // 2. Reshape for Multi-Head Attention
        // [B, SeqLen, heads * headDim] -> [B, heads, SeqLen, headDim]
        // Prefer SDPA: materializing [heads, 3072, 3072] scores (≈600MB/layer) via matmul
        // routinely hangs Apple GPUs when the full 16-layer LRM graph is fused.
        let qReshaped = q.reshaped([B, targetLen, heads, headDim]).transposed(axes: [0, 2, 1, 3])
        let kReshaped = k.reshaped([B, sourceLen, heads, headDim]).transposed(axes: [0, 2, 1, 3])
        let vReshaped = v.reshaped([B, sourceLen, heads, headDim]).transposed(axes: [0, 2, 1, 3])
        
        // 3–4. Scaled dot-product attention (scale applied inside; output [B, heads, T, D])
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: qReshaped,
            keys: kReshaped,
            values: vReshaped,
            scale: scale,
            mask: .none
        )
        
        // 5. Reshape back: [B, heads, SeqLen, headDim] -> [B, SeqLen, heads * headDim]
        let out = attnOut.transposed(axes: [0, 2, 1, 3]).reshaped([B, targetLen, heads * headDim])
        
        // 6. Output projection
        return toOut(out)
    }
}
