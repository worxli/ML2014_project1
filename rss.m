function [ err ] = rss( X,y )

% Linear regression
b = cgs((X'*X), X'*y);

err = norm(X*b - y);


end

