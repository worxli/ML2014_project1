function [ beta ] = regression( Xtr,y,lambda )
 beta = inv(Xtr'*Xtr+lambda*eye(size(Xtr,2)))*Xtr'*y;
        %beta = ridge(y(ind ~= i),Xtr,k,0);
        %beta=lasso(Xtr,y(ind ~= i),'Lambda',k);

end

