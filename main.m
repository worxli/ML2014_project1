clear;

%% read data
training = csvread('training.csv');
%testdata = csvread('testing.csv');
validation = csvread('validation.csv');

%%  normalization
%range = [6 128 72 72 120 14 7 31000 768 24 960 960 7488 27]./2;
%mean = [5 96 44 44 100 9 4.5 16500 635 20 544 544 4256 22.5];

%averagedata = data - repmat(mean,333,1);
%x = bsxfun(@rdivide, averagedata, range);
%x = [x y]; 

MEAN = mean(training);
STD = std(training);
averagedata = training-repmat(MEAN,size(training,1),1);
normdata = bsxfun(@rdivide, averagedata, STD);

%normdata = [normdata; normdata];

X = normdata(:,1:end-1);
y = normdata(:,end);
X = [ones(size(X)) X X.^2 X.^5 X.^10];


%% ridge regression

%possible lambdas
lambda = exp(-10:0.5:10);%0:0.1:10;%exp(-10:1:10);

%kfold default=10
kfold = 10;

candidates = [];
%iterate over all lambdas
for k=lambda

    err = [];
    betas = [];
    candidate = [];
    ind = crossvalind('Kfold', size(X,1), kfold);
   
    % do kfold crossvalidation for each lambda
    for i = 1:kfold
        Xts = X(ind == i, :);
        Xtr = X(ind ~= i, :);
        
        % closed form solution -> may be replaced by gradient descent
        beta = inv(Xtr'*Xtr+k*eye(size(Xtr,2)))*Xtr'*y(ind ~= i);
        betas = [betas; beta'];
        
        %estimate current lambda's error
        curerr = norm(Xts*beta - y(ind == i));
        err = [err curerr ];
    end
    
    [val,ind] = min(err);
    candidate = [val betas(ind,:)];
   
    candidates = [candidates; candidate]; 
    
end

%get index for lambda with lowest error
[val, ind] = min(candidates(:,1));

%calculate beta with chosen lambda
ridgebeta = inv(X'*X+lambda(ind)*eye(size(X,2)))*X'*y;

%calculate and show error for beta estimate
ridgeerr = X*ridgebeta-y;
norm(ridgeerr)

%% test on validation set

% normalize validation data
averagedata = validation-repmat(MEAN(1:end-1),size(validation,1),1);
normdata = bsxfun(@rdivide, averagedata, STD(1:end-1));

%model definition
normdata = [ones(size(normdata)) normdata normdata.^2 normdata.^5 normdata.^10];

% calculate prediction and un-normalize
prediction = normdata*ridgebeta;
unnormpred = bsxfun(@times, prediction, STD(end));
preddata = unnormpred+repmat(MEAN(end),size(validation,1),1);

%% write to csv file for submission
csvwrite('validationsetresult.csv', preddata);







