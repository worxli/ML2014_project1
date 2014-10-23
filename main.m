clear;

%% read data
training = csvread('training.csv');
%testdata = csvread('testing.csv');
validation = csvread('validation.csv');

%%  normalization

MEAN = mean(training);
STD = std(training);
averagedata = training-repmat(MEAN,size(training,1),1);
normdata = bsxfun(@rdivide, averagedata, STD);

%Features
X = normdata(:,1:end-1);
y = normdata(:,end);
X= getFeatures(X);

%% ridge regression

%possible lambdas
%lambda = exp(-1:0.1:5);
lambda = 0.1:0.1:10;%exp(-10:1:10);

%kfold default=10
kfold = 10;

candidates = [];
errs=[];
%iterate over all lambdas
for k=lambda

    err = 0;
    betas = [];
    candidate = [];
    ind = crossvalind('Kfold', size(X,1), kfold);
   
    % do kfold crossvalidation for each lambda
    for i = 1:kfold
        Xts = X(ind == i, :);
        Xtr = X(ind ~= i, :);
        
        % closed form solution -> may be replaced by gradient descent
        beta= regression(Xtr,y(ind ~= i),k);
        betas = [betas; beta'];
        
        %estimate current lambda's error
        curerr = norm(Xts*beta - y(ind == i));
        err = err+curerr;
    end
    errs=[errs err];
end

%get index for lambda with lowest error
[val,ind]=min(errs);

%calculate beta with chosen lambda
ridgebeta = regression(X,y,lambda(ind));

%calculate and show error for beta estimate
ridgeerr = X*ridgebeta-y;
disp(['Error is ' num2str(norm(ridgeerr)) ' w Lambda ' num2str(lambda(ind))]);
%% test on validation set

% normalize validation data
averagedata = validation-repmat(MEAN(1:end-1),size(validation,1),1);
normdata = bsxfun(@rdivide, averagedata, STD(1:end-1));

%model definition
normdata = getFeatures(normdata);

% calculate prediction and un-normalize
prediction = normdata*ridgebeta;
unnormpred = bsxfun(@times, prediction, STD(end));
preddata = unnormpred+repmat(MEAN(end),size(validation,1),1);

%% write to csv file for submission
csvwrite('validationsetresult.csv', preddata);







