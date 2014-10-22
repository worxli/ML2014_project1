%% read data
training = csvread('training.csv');
%testdata = csvread('testing.csv');
validation = csvread('validation.csv');

data = training(:,1:end-1);
y = training(:,end);

no_buckets = 10;
bucketsize = floor(size(training,1)/no_buckets); %% don't care about the last three

%%  normalization
range = [6 128 72 72 120 14 7 31000 768 24 960 960 7488 27]./2;
mean = [5 96 44 44 100 9 4.5 16500 635 20 544 544 4256 22.5];

averagedata = data - repmat(mean,333,1);
x = bsxfun(@rdivide, averagedata, range);
x = [x y]; 


%% ridge regression

% cross validation for lambda
err = [];
err2 = [];
betas = [];
for i=1:no_buckets
    
    %validation set indices
    valSetIndex = [(i-1)*bucketsize+1, i*bucketsize];
    
    %validation set
    V = x(valSetIndex(1):valSetIndex(2),:);
    Vy = V(:,end);
    V = V(:,1:end-1);
    
    %training set w/o the validation set
    X = [ones(size(x,1)-size(V,1),1) [x(1:valSetIndex(1)-1,:) ; x(valSetIndex(2)+1:end,:) ]];
    y = X(:,end);
    X = X(:,1:end-1);
    
    %beta = inv(X'*X+10^(i+3)*eye(size(X,2)))*X'*y;
    beta = ridge(y,X,2^-i)
    
    
    betas(i,:) = [beta];
    err = [err min(abs(V*beta(1:end-1) - Vy))];
    err2 = [err2 norm(V*beta(1:end-1) - Vy)];
    
    
end

averagedata = validation - repmat(mean,333,1);
normval = bsxfun(@rdivide, averagedata, range);

csvwrite('validationsetresult.csv', normval*betas(5,1:end-1)');


plot(err);
figure;
plot(err2);


%% plotting