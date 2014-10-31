function [ F ] = getFeatures(X)

    MIX = getMixedTerm(X);
    %F = [ones(size(X,1),1) log(X) 1./X sqrt(X) X X.^2 X.^3 X.^5];
    %F = [ones(size(X,1),1) X X.^2 X.^3 X.^4 X.^5 abs(sqrt(X)) 1./X abs(log(X))];
    
    F = x2fx(X, 'quadratic');
    
    F = [F X(:,1)./X(:,14) exp(X) abs(log(X)) abs(sqrt(X)) 1./X MIX];

end