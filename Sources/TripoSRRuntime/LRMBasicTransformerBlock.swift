import Foundation
import MLX
import MLXNN

public class LRMBasicTransformerBlock: Module {
    public let norm1: LayerNorm
    public let attn1: LRMAttention
    
    public let norm2: LayerNorm?
    public let attn2: LRMAttention?
    
    public let norm3: LayerNorm
    public let ff: LRMFeedForward
    
    public init(
        dim: Int,
        numAttentionHeads: Int,
        attentionHeadDim: Int,
        crossAttentionDim: Int? = nil
    ) {
        // 1. Self-Attention
        self.norm1 = LayerNorm(dimensions: dim, eps: 1e-5)
        self.attn1 = LRMAttention(
            queryDim: dim,
            heads: numAttentionHeads,
            dimHead: attentionHeadDim,
            bias: false
        )
        
        // 2. Cross-Attention
        if let crossDim = crossAttentionDim {
            self.norm2 = LayerNorm(dimensions: dim, eps: 1e-5)
            self.attn2 = LRMAttention(
                queryDim: dim,
                crossAttentionDim: crossDim,
                heads: numAttentionHeads,
                dimHead: attentionHeadDim,
                bias: false
            )
        } else {
            self.norm2 = nil
            self.attn2 = nil
        }
        
        // 3. Feed-Forward
        self.norm3 = LayerNorm(dimensions: dim, eps: 1e-5)
        self.ff = LRMFeedForward(dim: dim)
    }
    
    public func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        // 1. Self-Attention
        var normHiddenStates = norm1(hiddenStates)
        let attn1Output = attn1(normHiddenStates)
        var out = hiddenStates + attn1Output
        
        // 2. Cross-Attention
        if let attn2 = attn2, let norm2 = norm2 {
            normHiddenStates = norm2(out)
            let attn2Output = attn2(normHiddenStates, encoderHiddenStates: encoderHiddenStates)
            out = out + attn2Output
        }
        
        // 3. Feed-Forward
        normHiddenStates = norm3(out)
        let ffOutput = ff(normHiddenStates)
        out = out + ffOutput
        
        return out
    }
}
