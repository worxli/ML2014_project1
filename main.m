%% read data
training = csvread('training.csv');
%testdata = csvread('testing.csv');
%validation = csvread('validation.csv');

data = training(:,1:end-1);
y = training(:,end);

no_buckets = 10;
bucketsize = floor(size(training,1)/no_buckets) %% don't care about the last three

%%  normalization

range = [6 128 72 72 120 14 7 31000 768 24 960 960 7488 27]./2;
mean = [5 96 44 44 100 9 4.5 16500 635 20 544 544 4256 22.5];

averagedata = data - repmat(mean,333,1);
x = bsxfun(@rdivide, averagedata, range);
x = [x y]; 


%% ridge regression

for i=1:no_buckets
    
    bucketsize_inner = floor((bucketsize*no_buckets-bucketsize)/no_buckets);
    for j=1:bucketsize
        
        X = [ones(size(data,1),1) x];
        
        
        
    end
    
end



%% plotting