function [ val, ind ] = crossValidation( Xt, y, lambdas, kfold )

errs = [];
 
%iterate over all lambdas
for k=lambdas

    err = 0;
    ind = crossvalind('Kfold', size(Xt,1), kfold);

    % do kfold crossvalidation for each lambda
    for i = 1:kfold
        Xts = Xt(ind == i, :);
        Xtr = Xt(ind ~= i, :);

        % closed form solution -> may be replaced by gradient descent
        beta = regression(Xtr,y(ind ~= i),k);

        %estimate current lambda's error
        curerr = norm(Xts*beta - y(ind == i));
        err = err + curerr;
    end

    errs = [errs err];
end

%plot(lambda,errs);

[val, ind] = min(errs);

end

