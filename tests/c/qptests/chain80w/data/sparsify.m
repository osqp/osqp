function Asp = sparsify(A, tol)
    Asp = zeros(size(A));
    for i=1:size(A,1)
        for j=1:size(A,2)
            if abs(A(i,j))<= tol
                Asp(i,j) = 0;
            else
                Asp(i,j) = A(i,j);
            end
        end
    end
    Asp = sparse(Asp);
end
