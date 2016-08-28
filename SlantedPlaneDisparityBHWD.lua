local SlantedPlaneDisparity, parent = torch.class('nn.SlantedPlaneDisparityBHWD', 'nn.Module')

function SlantedPlaneDisparity:__init(height, width)
   parent.__init(self)
   assert(height > 1)
   assert(width > 1)
   self.height = height
   self.width = width
   
   self.baseGrid = torch.Tensor(height, width, 3)
   for i=1,self.height do
      self.baseGrid:select(3,1):select(1,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   for j=1,self.width do
      self.baseGrid:select(3,2):select(2,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end
   self.baseGrid:select(3,3):fill(1)
   self.batchGrid = torch.Tensor(1, height, width, 3):copy(self.baseGrid)
end

function SlantedPlaneDisparity:updateOutput(_PerPixelSlant)

   local PerPixelSlant = _PerPixelSlant

   --[[if _transformMatrix:nDimension()==2 then
      transformMatrix = addOuterDim(_transformMatrix)
   else
      transformMatrix = _transformMatrix
   end]]--

   assert(PerPixelSlant:nDimension()==4
	  and PerPixelSlant:size(2)==self.height
          and PerPixelSlant:size(3)==self.width
          and PerPixelSlant:size(4)==3
          , 'please input affine per-pixel transformations (bxhxwx3)')

   local batchsize = PerPixelSlant:size(1)
   
   if self.batchGrid:size(1) ~= batchsize then

      self.batchGrid:resize(batchsize, self.height, self.width, 3)

      for i=1,batchsize do
         self.batchGrid:select(1,i):copy(self.baseGrid)
      end

   end

   self.output:resize(batchsize, self.height, self.width, 1)

   self.output:select(4,1):copy(torch.sum(torch.cmul(PerPixelSlant, self.batchGrid),4))

   return self.output

end

function SlantedPlaneDisparity:updateGradInput(_PerPixelSlant, _gradGrid)

   local PerPixelSlant, gradGrid

   transformMatrix = _transformMatrix
   gradGrid = _gradGrid

   local batchsize = PerPixelSlant:size(1)

   self.gradInput:resizeAs(_PerPixelSlant):zero()

   local L_x = torch.cmul(_gradGrid, self.batchGrid:select(4,1))	
   local L_y = torch.cmul(_gradGrid, self.batchGrid:select(4,2))	

   self.gradInput:select(4,1):copy(L_x)
   self.gradInput:select(4,2):copy(L_y)
   
   return self.gradInput

end
