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

X = normdata(:,1:end-1);
y = normdata(:,end);
Xt = [ones(size(X,1),1) X X.^2];


%% ridge regression

%possible lambdas
lambda = exp(-2:0.1:10);%0:0.1:10;%exp(-10:1:10);

%kfold default=10
kfold = 10;

errs = [];

%iterate over all lambdas
for k=lambda

    err = 0;
    ind = crossvalind('Kfold', size(Xt,1), kfold);
   
    % do kfold crossvalidation for each lambda
    for i = 1:kfold
        Xts = Xt(ind == i, :);
        Xtr = Xt(ind ~= i, :);
        
        % closed form solution -> may be replaced by gradient descent
        beta = inv(Xtr'*Xtr+k*eye(size(Xtr,2)))*Xtr'*y(ind ~= i);
        
        %estimate current lambda's error
        curerr = norm(Xts*beta - y(ind == i));
        err = err+curerr;
    end
    
    errs = [errs err];
    
end

%get index for lambda with lowest error
[val, ind] = min(errs);
disp(['Cumulative prediction error for lambda ' num2str(lambda(ind)) ' is: ' num2str(val)]);

%calculate beta with chosen lambda
ridgebeta = inv(Xt'*Xt+lambda(ind)*eye(size(Xt,2)))*Xt'*y;

%calculate and show error for beta estimate
ridgeerr = Xt*ridgebeta-y;
norm(ridgeerr)

%% test on validation set

% normalize validation data
averagedata = validation-repmat(MEAN(1:end-1),size(validation,1),1);
normdata = bsxfun(@rdivide, averagedata, STD(1:end-1));

%model definition
normdata = [ones(size(normdata,1),1) normdata normdata.^2];

% calculate prediction and un-normalize
prediction = normdata*ridgebeta;
unnormpred = bsxfun(@times, prediction, STD(end));
preddata = unnormpred+repmat(MEAN(end),size(validation,1),1);

%% write to csv file for submission
csvwrite('validationsetresult.csv', preddata);







