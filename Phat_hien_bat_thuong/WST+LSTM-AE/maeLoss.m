function loss = maeLoss(Y,T)
    loss = mean(abs(Y-T),"all");
end