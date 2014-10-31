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

%% Maybe use sequentials

%% ridge regression

%possible lambdas
%lambda = exp(-1:0.1:5);
lambdas = 1:1:10;

%kfold default=10
kfold = 5;

featurevec = 1:size(Xt,2);
errs = [];
indices = [];
featuresvec = [];
parfor i = 3:size(Xt,2)
   
    features = nchoosek(featurevec,i);
    if size(features,2)>1
        for j = 1:size(features,1)
            [err, index] = crossValidation(Xt(:,features(j,:)),y,lambdas,kfold);
            errs = [errs err];
            indices = [indices index];
            emptyvec = zeros(length(featurevec) ,1);
            emptyvec(1:length(features(j,:))) = features(j,:);
            featuresvec = [featuresvec; emptyvec'];
        end
    else
        [err, index] = crossValidation(Xt(:,features),y,lambdas,kfold); 
        errs = [errs err];
        indices = [indices index];
        emptyvec = zeros(length(featurevec) ,1);
        emptyvec(1:length(features)) = features;
        featuresvec = [featuresvec; emptyvec'];
    end
    
end

%plot(errs);

%get index for lambda with lowest error
[val, ind] = min(errs);
lambda = lambdas(indices(ind));
disp(['Prediction error for lambda ' num2str(lambda) ' is: ' num2str(val) ' (chosen lambda), MODEL ERROR: ' num2str(sum(errs))]);

selectedFeatures = featuresvec(ind,:)
selectedFeatures(selectedFeatures==0) = [];

%calculate beta with chosen lambda
ridgebeta = regression(Xt(:,selectedFeatures),y,lambda);

%calculate and show error for beta estimate
ridgeerr = Xt(:,selectedFeatures)*ridgebeta-y;
disp(['Error on training data: ' num2str(norm(ridgeerr))]);

%% test on validation set

% normalize validation data
averagedata = validation-repmat(MEAN(1:end-1),size(validation,1),1);
normdata = bsxfun(@rdivide, averagedata, STD(1:end-1));

%model definition
normdata = getFeatures(normdata);
normdata = normdata(:, selectedFeatures);

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
normdata = normdata(:, selectedFeatures);

% calculate prediction and un-normalize
prediction = normdata*ridgebeta;
unnormpred = bsxfun(@times, prediction, STD(end));
preddata = unnormpred+repmat(MEAN(end),size(testdata,1),1);

%% write to csv file for submission
csvwrite('testsetresult.csv', preddata);







