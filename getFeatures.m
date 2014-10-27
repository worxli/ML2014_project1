function [ F ] = getFeatures(X)

    MIX = getMixedTerm(X);
    F = [ones(size(X,1),1) log(X) 1./X sqrt(X) X X.^2 X.^3 X.^5];

end