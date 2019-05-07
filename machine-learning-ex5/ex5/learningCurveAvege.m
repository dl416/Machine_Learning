function [error_train, error_val] = ...
    learningCurveAvege(X, y, Xval, yval, lambda)

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

max_iter = 20;

for i = 1:m
    err_train = 0;
    err_val = 0;
    parfor j = 1:max_iter
        index1 = randperm(m,i);
        Xtmp = X(index1, :);
        ytmp = y(index1, :);
        index2 = randperm(m,i);
        Xvaltmp = Xval(index2, :);
        yvaltmp = yval(index2, :);

        [theta] = trainLinearReg([ones(i, 1) Xtmp], ytmp, lambda);
        err_train = err_train + linearRegCostFunction([ones(i, 1), Xtmp], ytmp, theta, 0);
        err_val = err_val + linearRegCostFunction([ones(i, 1), Xvaltmp], yvaltmp, theta, 0);
    end
    error_train(i,1) = err_train./max_iter;
    error_val(i,1) = err_val./max_iter;
end

end
