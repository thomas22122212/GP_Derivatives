% Standard Squared Exponential (SE) kernel
% Input
%     X: training points
%     hyp: hyperparameters
%     XX: testing points (optional)
% Output
%     K: dense kernel matrix
%     dKhyp: kernel matrix derivatives w.r.t. hyperparameters

function [K, dKhyp] = se_kernel(X, hyp, XX)
    if nargin == 2
        XX = X;
    end

    ell = exp(hyp.cov(1)); 
    s = exp(hyp.cov(2));

    D = dis(XX,X);      %Replacement
    K = s^2*exp(-D.^2/(2*ell^2));

    if nargout == 2
        dKhyp = {1/ell^2 * (D.^2 .* K), 2*K};
    end
end

function d = dis(X,Y) %Replacement for pdist2(Y,X) to allow symbolic derivatives
    if (size(X,2) ~= size(Y,2))
        error('Dimension d inconsistent in X(n,d) and Y(m,d)');
    end
    if (class(X) == "sym" || class(Y) == "sym")
        syms temp [size(X,1) size(Y,1)];
	    for i=1:size(X,1)
	        for j=1:size(Y,1)
	            sum = 0;
	            for y = 1:size(X,2)
	                sum = sum + (X(i,y)-Y(j,y))^2;
	            end
	            temp(i,j) = sum^(1/2);
	        end
	    end
	    d = temp;
    else 
        d = pdist2(X, Y);
    end
end
