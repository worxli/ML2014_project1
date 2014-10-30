function [ F ] = getFeatures(X)

    %MIX = getMixedTerm(X);
    %F = [ones(size(X,1),1) log(X) 1./X sqrt(X) X X.^2 X.^3 X.^5];
    %F = [ones(size(X,1),1) X X.^2 X.^3 X.^4 X.^5 abs(sqrt(X)) 1./X abs(log(X))];
    
    %F = [ones(size(X,1),1) X X(:,14) X(:,3).^4 X(:,14).^3 abs(log(X(:,14))) X(:,4).^5 X(:,3).^3];
    %F = [ones(size(X,1),1) X(:,1).*X(:,3) X(:,1) X(:,3) X(:,4) X(:,6) X(:,14) X(:,1).^2 X(:,3).^2 X(:,4).^2 X(:,14).^2 X(:,1).^3 X(:,14).^3 abs(sqrt(X(:,1))) abs(sqrt(X(:,12))) abs(log(X(:,14)))];
    %F = [ones(size(X,1),1) abs(log(X))];

    
    %F = x2fx(X, 'linear');
    %F = [F ones(size(X,1),1) abs(log(X))];
    
    %F = [F MIX X.^3 X.^4 abs(log(X)) 1./X abs(sqrt(X)) X.^-2];
   % F = [ones(size(X,1),1) X(:,14) X(:,14).^2 X X.^2 X.^3 X.^4];
   
   %F = [ones(size(X,1),1) X X.^2 X.^3 X.^4 X.^5 abs(sqrt(X)) abs(log(X)) 1./X];
   F = [ones(size(X,1),1) X(:,1:3)];
end