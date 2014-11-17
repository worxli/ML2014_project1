clear;

%% read data
training = csvread('training.csv');
testdata = csvread('testing.csv');
validation = csvread('validation.csv');

%%  normalization

MEAN = mean(training);
STD = std(training);

% Remove outliers
%training = removeOutliers(training, MEAN, STD);

averagedata = training-repmat(MEAN,size(training,1),1);
normdata = bsxfun(@rdivide, averagedata, STD);

%Features
X = normdata(:,1:end-1);
y = normdata(:,end);
Xt = getFeatures(X);
disp(['Size of feature space: ' num2str(size(Xt,2))]);

%% Maybe use sequentialfs
% Select good features with lasso
%[B, FitInfo] = lasso(Xt, y, 'CV', 10);

%% ridge regression

%possible lambdas
%lambda = exp(-1:0.1:5);
lambda = 0.001:0.001:0.02;

%kfold default=10
kfold = 10;

selectedFeatures = [];

oldError = 99999;
newError = 9999;
% Loop over feature space until enough found
while ((newError < oldError) && length(selectedFeatures) <= size(X,2))
    
    oldError = newError
    
    featureErrors = [];
    
    % For each feature compute the error for a random for a certain labmda
    for featIdx=1:size(Xt,2)

        % Only check new feature that is not already in selected features
        if (any(selectedFeatures == featIdx) == 0)
       
            % For all lambdas compute crossvalidation
            [errs, index] = crossValidation(Xt(:,[selectedFeatures featIdx]), y, lambda, kfold);
            
            % Store all errors for one feature
            featureErrors = [featureErrors min(errs)];
            
        else
            
            featureErrors = [featureErrors 999999];
        
        end
        
    end

    % Choose the feature with the smallest error
    [minFeatureError, idx] = min(featureErrors);

    newError = minFeatureError;
    
    % Store index of the selected feature
    selectedFeatures = [selectedFeatures idx];
    
end

selectedFeatures

% TODO: Xt(:,selectedFeatures) Run cross validation with all features
% And find best lambda
[errs, index] = crossValidation(Xt(:,selectedFeatures),y,lambda,kfold);

% All features selected:
% 226 15 28 229 231 260 243 267 112 42 115 5 78 227 53 2 51 103 20 275 32

% Plot lambda error for the final selected feature set
plot(lambda,errs);

%get index for lambda with lowest error
[val, ind] = min(errs);
disp(['Prediction error for lambda ' num2str(lambda(ind)) ' is: ' num2str(val) ' (chosen lambda), MODEL ERROR: ' num2str(sum(errs))]);

%calculate beta with chosen lambda
ridgebeta = regression(Xt(:,selectedFeatures),y,lambda(ind));

%calculate and show error for beta estimate
ridgeerr = Xt(:,selectedFeatures)*ridgebeta-y;
disp(['Error on training data: ' num2str(norm(ridgeerr)) ]);

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

% normalize test data
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
