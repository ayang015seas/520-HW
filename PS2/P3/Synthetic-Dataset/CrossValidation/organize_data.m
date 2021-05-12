

for i = 1:5
    data = dlmread(strcat("./Fold", int2str(i), "/cv-test.txt"));
    data_x = data(:, 1:end-1);
    data_y = data(:, end);
    dlmwrite(strcat("./Fold", int2str(i), "/X.txt"), data_x);
    dlmwrite(strcat("./Fold", int2str(i), "/y.txt"), data_y);
end