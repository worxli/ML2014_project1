function [ F ] = getFeatures(X)
MIX = getMixedTerm(X);
F = [ones(size(X,1),1) MIX X X.^2 X.^3];
end


