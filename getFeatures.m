function [ F ] = getFeatures(X)

    %MIX = getMixedTerm(X);
    %F = [ones(size(X,1),1) log(X) 1./X sqrt(X) X X.^2 X.^3 X.^5];
    %F = [ones(size(X,1),1) 1./X X X.^5];
    
    %F = x2fx(X, 'quadratic');
    
    %F = [F MIX X.^3 X.^4 abs(log(X)) 1./X abs(sqrt(X)) X.^-2];
    F = X;
end