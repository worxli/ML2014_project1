function [ MIX ] = getMixedTerm( X)
MIX = [];
% for i=1:14
%    for j=i:14
%        MIX = [MIX X(:,i).*X(:,j)];
%    end
% end
 MIX = [MIX X.^-3 X.^-1]; 
end
