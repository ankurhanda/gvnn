local Disparity1D, parent = torch.class('nn.Disparity1DBHWD', 'nn.Module')

--[[
   Disparity1D(height, width) :
   Disparity1D:updateOutput(transformMatrix)
   Disparity1D:updateGradInput(transformMatrix, gradGrids)

   Disparity1D will take 2x3 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.

   AffineGridGenerator 
   - takes (B,2,3)-shaped transform matrices as input (B=batch).
   - outputs a grid in BHWD layout, that can be used directly with BilinearSamplerBHWD
   - initialization of the previous layer should biased towards the identity transform :
      | 1  0  0 |
      | 0  1  0 |

   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function Disparity1D:__init(height, width)
    
   parent.__init(self)
   assert(height > 1)
   assert(width > 1)
   self.height = height
   self.width = width
   
   self.baseGrid = torch.Tensor(height, width, 2)

   for i=1,self.height do
      self.baseGrid:select(2,2):select(1,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   for j=1,self.width do
      self.baseGrid:select(2,1):select(2,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end

end

function Disparity1D:updateOutput(disparity1D)

   local current_disparity  = disparity1D

   assert(current_disparity:nDimension()==4
	  and current_disparity:size(2)==self.height
          and current_disparity:size(3)==self.width
          and current_disparity:size(4)==1
          , 'please input affine per-pixel transformations (bxhxwx2)')

   local batchsize = current_disparity:size(1)
   
   if self.batchGrid:size(1) ~= batchsize then

      self.batchGrid:resize(batchsize, self.height, self.width, 2)

      for i=1,batchsize do
         self.batchGrid:select(1,i):copy(self.baseGrid)
      end

   end

   self.output:resize(batchsize, self.height, self.width, 2)
   self.output:select(4,1):copy(torch.add(self.baseGrid:select(2,1),current_disparity))
   
   return self.output

end

function Disparity1D:updateGradInput(disparity1D, _gradGrid)

   self.gradInput:resize(disparity1D:size(1), self.height, self.width, 2):zero():typeAs(disparity1D)
   self.gradInput:select(4,1):copy(_gradGrid:select(4,1))
   self.gradInput:select(4,2):zero()

   return self.gradInput

end
