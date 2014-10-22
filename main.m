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

X = normdata(:,1:end-1);
y = normdata(:,end);


%% ridge regression

lambda = 0:0.05:10;%exp(-10:1:10);
kfold = 10;

candidates = [];
for k=lambda

    err = [];
    betas = [];
    candidate = [];
    Xt = [ones(size(X,1),1) X];
    ind = crossvalind('Kfold', size(X,1), kfold);
   
    for i = 1:kfold
        Xts = Xt(ind == i, :);
        Xtr = Xt(ind ~= i, :);
        
        beta = inv(Xtr'*Xtr+k*eye(size(Xtr,2)))*Xtr'*y(ind ~= i);
        betas = [betas; beta'];
        x = 1:size(Xts,1);
        %plot(x, Xts*beta, x, y(ind == i));
        %pause(0.5);
        err = [err  norm(Xts*beta - y(ind == i))];
    end
    
    %plot(err);
    %hold on;
    [val,ind] = min(err);
    candidate = [val betas(ind,:)];
   
    candidates = [candidates; candidate]; 
    
end

[val, ind] = min(candidates(:,1));
beta = candidates(ind,2:end);

ridgebeta = inv(X'*X+lambda(ind)*eye(size(X,2)))*X'*y;



plot(1:333, X*ridgebeta, 1:333, y);

% test on validation set

averagedata = validation-repmat(MEAN(1:end-1),size(validation,1),1);
normdata = bsxfun(@rdivide, averagedata, STD(1:end-1));

prediction = normdata*ridgbeta;
unnormpred = bsxfun(@times, prediction, STD(end));
preddata = unnormpred+repmat(MEAN(end),size(validation,1),1);

csvwrite('validationsetresult.csv', preddata);











