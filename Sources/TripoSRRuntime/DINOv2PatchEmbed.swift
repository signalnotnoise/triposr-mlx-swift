import Foundation
import MLX
import MLXNN

public class DINOv2PatchEmbed: Module {
    public let patchSize: Int
    public let embedDim: Int
    public let proj: Conv2d
    
    public init(patchSize: Int = 14, inChannels: Int = 3, embedDim: Int = 768) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        // In MLX, Conv2d takes (kernelSize, stride). We use stride = patchSize.
        self.proj = Conv2d(
            inputChannels: inChannels, 
            outputChannels: embedDim, 
            kernelSize: [patchSize, patchSize], 
            stride: [patchSize, patchSize], 
            padding: [0, 0]
        )
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x expected shape: [B, H, W, C] (MLX native format)
        let B = x.shape[0]
        
        // 1. Convolution: [B, H, W, C] -> [B, H/P, W/P, embedDim]
        var out = proj(x)
        
        let H_out = out.shape[1]
        let W_out = out.shape[2]
        let seqLen = H_out * W_out
        
        // 2. Flatten spatial dimensions: [B, H/P * W/P, embedDim]
        out = out.reshaped([B, seqLen, embedDim])
        return out
    }
}
