import Foundation
import MLX
import MLXNN

public class DINOv2Layer: Module {
    public let norm1: LayerNorm
    public let attn: DINOv2Attention
    public let ls1: MLXArray?
    
    public let norm2: LayerNorm
    public let mlp: DINOv2MLP
    public let ls2: MLXArray?
    
    /// DINO ViT-B/16 (facebook/dino-vitb16) has no layer-scale; keep default `nil`.
    public init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0, layerScaleInitValue: Float? = nil) {
        self.norm1 = LayerNorm(dimensions: dim, eps: 1e-6)
        self.attn = DINOv2Attention(dim: dim, numHeads: numHeads)
        
        self.norm2 = LayerNorm(dimensions: dim, eps: 1e-6)
        self.mlp = DINOv2MLP(inFeatures: dim, hiddenFeatures: Int(Float(dim) * mlpRatio), outFeatures: dim)
        
        if let initVal = layerScaleInitValue {
            self.ls1 = MLXArray.ones([dim]) * initVal
            self.ls2 = MLXArray.ones([dim]) * initVal
        } else {
            self.ls1 = nil
            self.ls2 = nil
        }
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var out = attn(norm1(x))
        if let ls1 = ls1 {
            out = out * ls1
        }
        var xOut = residual + out
        
        residual = xOut
        out = mlp(norm2(xOut))
        if let ls2 = ls2 {
            out = out * ls2
        }
        xOut = residual + out
        
        return xOut
    }
}
