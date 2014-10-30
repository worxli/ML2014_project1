clear;

%% read data
training = csvread('training.csv');
testdata = csvread('testing.csv');
validation = csvread('validation.csv');

%%  normalization
MEAN = mean(training);
STD = std(training);

averagedata = training-repmat(MEAN,size(training,1),1);
normdata = bsxfun(@rdivide, averagedata, STD);

%Features
X = normdata(:,1:end-1);
y = normdata(:,end);
Xt = getFeatures(X);

%% ridge regression

%possible lambdas
%lambda = exp(-1:0.1:5);
lambda = 0.1:0.1:20;

%kfold default=10
kfold = 10;

%iterate over all lambdas
errs = [];
for k=lambda

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
        err = err+curerr;
    end

    errs = [errs err];
end

plot(lambda,errs);

%get index for lambda with lowest error
[val, ind] = min(errs);
disp(['Prediction error for lambda ' num2str(lambda(ind)) ' is: ' num2str(val) ' (chosen lambda), MODEL ERROR: ' num2str(sum(errs))]);

%calculate beta with chosen lambda
ridgebeta = regression(Xt,y,lambda(ind));

%calculate and show error for beta estimate
ridgeerr = Xt*ridgebeta-y;
disp(['Error on training data: ' num2str(norm(ridgeerr)) ]);

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

%% test on test set

% normalize validation data
averagedata = testdata-repmat(MEAN(1:end-1),size(testdata,1),1);
normdata = bsxfun(@rdivide, averagedata, STD(1:end-1));

%model definition
normdata = getFeatures(normdata);

% calculate prediction and un-normalize
prediction = normdata*ridgebeta;
unnormpred = bsxfun(@times, prediction, STD(end));
preddata = unnormpred+repmat(MEAN(end),size(testdata,1),1);

%% write to csv file for submission
csvwrite('testsetresult.csv', preddata);







