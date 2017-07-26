function convertProblemToMat(filename, Pdata, qdata, Adata, ldata, udata)
    Ptrip = load(Pdata);
    P = spconvert(Ptrip);
    q = load(qdata);
    l = load(ldata);
    u = load(udata);
    Atrip = load(Adata);
    A = spconvert(Atrip);
    
    % Dump matrices to .mat file
    save(filename, 'P', 'q', 'A', 'l', 'u');



end