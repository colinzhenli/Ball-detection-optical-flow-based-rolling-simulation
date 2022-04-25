function  output = imblend( source, mask, target, transparent )
%Source, mask, and target are the same size (as long as you do not remove
%the call to fiximages.m). You may want to use a flag for whether or not to
%treat the source object as 'transparent' (e.g. taking the max gradient
%rather than the source gradient).
source = py.numpy.array(source);
mask = py.numpy.array(mask);
target = py.numpy.array(target);
source = nparray2mat(source);
target = nparray2mat(target);
mask = nparray2mat(mask);
mask = im2gray(mask);
[sizeRow, sizeColumn] = size(mask);
source = [source(1,:, :); source];
target = [target(1,:, :); target];
source = [source, source(:,sizeColumn, :)];
target = [target, target(:,sizeColumn, :)];
source = [source; source(sizeRow+1, :, :)];
target = [target; target(sizeRow+1, :, :)];

maskUp = repmat(0, [1 sizeColumn]);
maskRight = repmat(0, [sizeRow+1 1]);
maskDown = repmat(0, [1 sizeColumn+1]);
mask = [maskUp; mask];
mask = [mask, maskRight];
mask = [mask; maskDown];
[sizeRow, sizeColumn] = size(mask);
sizeA = sizeRow * sizeColumn;

%get the matrix A
row = find(mask);
column = find(mask);
value = repmat(4, [size(row) 1]);
columnLeft = column - sizeRow;
columnRight = column + sizeRow;
columnUpper = column - 1;
columnLower = column + 1;
valueNeighbour = repmat(-1, [size(row) 1]);
row = [row; row; row; row; row];
column = [column; columnLeft; columnRight; columnUpper; columnLower];
value = [value; valueNeighbour; valueNeighbour; valueNeighbour; valueNeighbour];
rowNonmask = find(~mask);
columnNonmask = find(~mask);
valueNonmask = repmat(1, [size(rowNonmask) 1]);
row = [row; rowNonmask];
column = [column; columnNonmask];
value = [value; valueNonmask];
A = sparse(row, column, value, sizeA, sizeA);
%get the vector b for each channel
maskPixel = find(mask);
for i=1:3
    targetChannel = target(:,:,i);
    sourceChannel = source(:,:,i);
    %initialize b as the target image
    for index = 1:sizeA
        columnIndex = floor((index-1) / sizeRow) + 1;
         rowIndex = index - (columnIndex-1)*sizeRow;
        b(i, index) = targetChannel(rowIndex, columnIndex);
    end
    % compute the masked region of b
    for index = 1:size(maskPixel, 1)
        %get the subscripts for indices
        columnIndex = floor((maskPixel(index)-1) / sizeRow) + 1;
        rowIndex = maskPixel(index) - (columnIndex-1)*sizeRow;
        b(i, maskPixel(index)) = 4*sourceChannel(rowIndex, columnIndex) - sourceChannel(rowIndex-1, columnIndex) - sourceChannel(rowIndex+1, columnIndex) - sourceChannel(rowIndex, columnIndex-1) - sourceChannel(rowIndex, columnIndex+1);
    end
    bi = (b(i,:)).';
    xChannel = A \ bi;
    xChannel = xChannel.';
    x(:,:,i) = reshape(xChannel, [sizeRow, sizeColumn]);
end
%remove the added row and column
x(sizeRow,:,:) = [];
x(1,:,:) = [];
x(:,sizeColumn,:) = [];
output = x;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% As explained on the web page, we solve for output by setting up a large
% system of equations, in matrix form, which specifies the desired value or
% gradient or Laplacian (e.g.
% http://en.wikipedia.org/wiki/Discrete_Laplace_operator)

% The comments here will walk you through a conceptually simple way to set
% up the image blending, although it is not necessarily the most efficient
% formulation. 

% We will set up a system of equations A * x = b, where A has as many rows
% and columns as there are pixels in our images. Thus, a 300x200 image will
% lead to A being 60000 x 60000. 'x' is our output image (a single color
% channel of it) stretched out as a vector. 'b' contains two types of known 
% values:
%  (1) For rows of A which correspond to pixels that are not under the
%      mask, b will simply contain the already known value from 'target' 
%      and the row of A will be a row of an identity matrix. Basically, 
%      this is our system of equations saying "do nothing for the pixels we 
%      already know".
%  (2) For rows of A which correspond to pixels under the mask, we will
%      specify that the gradient (actually the discrete Laplacian) in the
%      output should equal the gradient in 'source', according to the final
%      equation in the webpage:
%         4*x(i,j) - x(i-1, j) - x(i+1, j) - x(i, j-1) - x(i, j+1) = 
%         4*s(i,j) - s(i-1, j) - s(i+1, j) - s(i, j-1) - s(i, j+1)
%      The right hand side are measurements from the source image. The left
%      hand side relates different (mostly) unknown pixels in the output
%      image. At a high level, for these rows in our system of equations we
%      are saying "For this pixel, I don't know its value, but I know that
%      its value relative to its neighbors should be the same as it was in
%      the source image".

% commands you may find useful: 
%   speye - With the simplest formulation, most rows of 'A' will be the
%      same as an identity matrix. So one strategy is to start with a
%      sparse identity matrix from speye and then add the necessary
%      values. This will be somewhat slow.
%   sparse - if you want your code to run quickly, compute the values and
%      indices for the non-zero entries in A and then construct 'A' with a
%      single call to 'sparse'.
%      Matlab documentation on what's going on under the hood with a sparse
%      matrix: www.mathworks.com/help/pdf_doc/otherdocs/simax.pdf
%   reshape - convert x back to an image with a single call.
%   sub2ind and ind2sub - how to find correspondence between rows of A and
%      pixels in the image. It's faster if you simply do the conversion
%      yourself, though.
%   see also find, sort, diff, cat, and spy


