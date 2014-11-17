function [ MIX ] = getMixedTerm( X)

MIX = [];
for i=1:14
   for j=i:14
       MIX = [MIX X(:,i)./X(:,j) X(:,i).*X(:,j) X(:,j)./X(:,i)];
   end
end

end

