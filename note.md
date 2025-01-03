
## MM

AOI: 0.757

|配置|ALL|LD|AD|QP|LM|Gap|DM|
|---|---|---|---|---|---|---|---|
|CFG:cascade_ml1|0.757|0.938|0.970|0.831|0.614|0.649|0.648|

precision 0.908, recall 0.915, f1 0.9115572436245253

CFPS: 
0.92007
|配置|ALL|BW|HQ|RGBBQ|BMDBQ|BMGBQ|PSZFH|SC|FBQ|PSZQS|
|---|---|---|---|---|---|---|---|---|---|---|
|CFG:cascade_ml1|0.890|0.922|0.951|0.970|0.948|0.975|0.824|0.851|1.000|0.868|

total test nr 1882, precision 92.554, recall 91.467, f1 92.007


CFOC: 0.976

: total test nr 446, precision 96.086, recall 96.275, f1 96.180


Per classes
|配置|ALL|BW|HQ|RGBBQ|BMDBQ|BMGBQ|SC|FBQ|PO|
|---|---|---|---|---|---|---|---|---|---|
|CFG:cascade_ml1|0.976|0.973|0.980|0.999|0.970|0.940|N.A.|1.000|1.000|

CFB: 0.973
: total test nr 509, precision 95.922, recall 97.351, f1 96.631


|配置|ALL|BW|HQ|RGBBQ|BMDBQ|BMGBQ|SC|FBQ|
|---|---|---|---|---|---|---|---|---|
|CFG:cascade_ml1|0.973|0.954|0.997|0.980|0.932|0.990|N.A.|1.000|

CFG: 0.941
: total test nr 706, precision 93.702, recall 96.429, f1 95.046

|配置|mAP|mAP@.50IOU|mAP@.75IOU|mAP (small)|mAP (medium)|mAP (large)|AR@1|AR@10|AR@100|AR@100 (small)|AR@100 (medium)|AR@100 (large)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||0.563|0.817|0.666|0.657|0.453|-1.000|0.596|0.629|0.629|0.734|0.468|-1.000|
Per classes
|配置|ALL|BW|HQ|RGBBQ|BMDBQ|BMGBQ|SC|FBQ|
|---|---|---|---|---|---|---|---|---|
|CFG:cascade_ml1|0.817|0.941|0.961|0.980|0.905|0.970|0.000|1.000|


CFR:
: total test nr 519, precision 92.691, recall 95.666, f1 94.155

Per classes
|配置|ALL|BW|HQ|RGBBQ|BMDBQ|BMGBQ|SC|FBQ|
|---|---|---|---|---|---|---|---|---|
|CFG:cascade_ml1|0.961|0.962|0.964|0.952|0.986|0.959|0.930|1.000|

CFBM:
: total test nr 757, precision 91.725, recall 97.149, f1 94.359

|配置|ALL|BW|HQ|BMDBQ|BMGBQ|SC|FBQ|FB|
|---|---|---|---|---|---|---|---|---|
|CFG:cascade_ml1|0.979|0.976|0.947|0.987|0.990|1.000|1.000|0.967|


CFITO:
: total test nr 865, precision 92.683, recall 93.699, f1 93.188

Per classes
|配置|ALL|BW|HQ|SC|YW|FB|
|---|---|---|---|---|---|---|
|CFG:cascade_ml1|0.954|0.942|0.932|N.A.|N.A.|1.000|




## YOLO 

### AOI yolov8l
                   Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        916       1126      0.424      0.349      0.354      0.171
                    LD        194        207      0.701      0.804      0.788      0.413
                    AD        227        232      0.813      0.711       0.74      0.315
                    QP        194        244      0.702      0.545      0.547      0.275
                    LM        132        266      0.291    0.00752     0.0112    0.00697
                   Gap        137        141          0          0     0.0235    0.00943
                    DM         34         36     0.0359     0.0278     0.0131     0.0074

yolo11m
                   all        916       1126      0.579      0.493      0.505      0.228
                    LD        194        207      0.754      0.709      0.752      0.356
                    AD        227        232      0.851      0.909      0.873      0.384
                    QP        194        244       0.71      0.578      0.647      0.292
                    LM        132        266      0.269     0.0902     0.0745     0.0218
                   Gap        137        141      0.547      0.312      0.383      0.162
                    DM         34         36      0.343      0.361      0.299      0.153

yolo11l
                   all        916       1126      0.536      0.427      0.415      0.199
                    LD        194        207      0.722      0.787      0.771      0.412
                    AD        227        232      0.824      0.888      0.848      0.372
                    QP        194        244      0.411      0.631      0.585      0.292
                    LM        132        266     0.0416    0.00752    0.00269   0.000432
                   Gap        137        141      0.218      0.248      0.178      0.078
                    DM         34         36          1          0      0.103     0.0415
### CFPS yolov8s

                   all       1882       2215      0.738      0.774      0.806      0.422
                    BW        666        786      0.794      0.919       0.92      0.531
                    HQ        632        709      0.763       0.89        0.9       0.48
                 RGBBQ        268        290      0.835      0.921      0.921      0.471
                 BMDBQ        197        243      0.703      0.794      0.777      0.271
                 BMGBQ        102        107      0.766      0.888      0.898      0.435
                 PSZFH         35         44      0.738      0.727      0.754      0.294
                    SC          7          7      0.738      0.714      0.819       0.68
                   FBQ         24         24      0.891      0.917      0.908      0.524
                 PSZQS          3          5      0.415        0.2      0.354      0.117

yolov8l
                  all       1882       2215      0.751      0.778      0.801      0.425
                    BW        666        786      0.797      0.913      0.918      0.524
                    HQ        632        709      0.764      0.883      0.898      0.479
                 RGBBQ        268        290      0.845      0.921      0.922      0.479
                 BMDBQ        197        243      0.768       0.79      0.801      0.269
                 BMGBQ        102        107      0.842      0.888      0.896      0.461
                 PSZFH         35         44      0.714       0.68      0.718      0.285
                    SC          7          7      0.718      0.714      0.806      0.626
                   FBQ         24         24      0.896      0.917      0.912      0.523
                 PSZQS          3          5      0.413      0.295      0.336      0.177

yolo11l
                   all       1882       2215      0.744      0.764      0.787      0.417
                    BW        666        786       0.82      0.907      0.925      0.533
                    HQ        632        709      0.793      0.883      0.897      0.484
                 RGBBQ        268        290      0.842      0.921      0.928       0.49
                 BMDBQ        197        243      0.787      0.811      0.813       0.28
                 BMGBQ        102        107       0.84      0.884      0.895      0.454
                 PSZFH         35         44       0.72      0.636      0.716      0.279
                    SC          7          7      0.684      0.714      0.771      0.603
                   FBQ         24         24      0.913      0.917      0.918       0.56
                 PSZQS          3          5      0.293        0.2       0.22     0.0718

### CFOC yolov8s

                   all        446        510      0.922      0.887      0.928      0.535
                    BW        154        168      0.876      0.923      0.915       0.53
                    HQ        148        162      0.951      0.951      0.982       0.53
                 RGBBQ         70         82      0.934      0.858       0.93      0.459
                 BMDBQ         35         41       0.93      0.652      0.813       0.32
                 BMGBQ         39         40      0.932      0.825      0.869      0.424
                   FBQ          4          4      0.881          1      0.995      0.559
                    PO         13         13      0.953          1      0.995      0.922
yolov8l
                   all        446        510      0.893       0.91      0.943      0.539
                    BW        154        168      0.877       0.93       0.94      0.531
                    HQ        148        162      0.931      0.957      0.972      0.545
                 RGBBQ         70         82      0.938      0.924       0.97      0.481
                 BMDBQ         35         41      0.909      0.731      0.843      0.334
                 BMGBQ         39         40       0.84      0.825      0.883      0.413
                   FBQ          4          4      0.873          1      0.995      0.595
                    PO         13         13      0.886          1      0.995      0.876
yolov8l 300 epoch
                   all        446        510      0.891      0.925      0.936      0.554
                    BW        154        168       0.86      0.951      0.935      0.533
                    HQ        148        162      0.915      0.957      0.982      0.553
                 RGBBQ         70         82      0.926      0.914      0.962      0.491
                 BMDBQ         35         41      0.825       0.78      0.801      0.349
                 BMGBQ         39         40      0.897      0.875      0.882      0.443
                   FBQ          4          4      0.867          1      0.995      0.585
                    PO         13         13      0.949          1      0.995      0.926

yolo11l
                   all        446        510      0.906      0.914      0.946      0.564
                    BW        154        168      0.876      0.905      0.905      0.508
                    HQ        148        162      0.953      0.951      0.987      0.549
                 RGBBQ         70         82      0.927       0.89      0.943      0.494
                 BMDBQ         35         41      0.764      0.829      0.863      0.376
                 BMGBQ         39         40      0.951      0.825      0.931      0.472
                   FBQ          4          4      0.872          1      0.995      0.585
                    PO         13         13          1          1      0.995      0.962

### CFB yolov8s
                   all        509        604      0.881       0.87      0.897      0.458
                    BW         88         97       0.91      0.939      0.939      0.525
                    HQ        125        136      0.894      0.912      0.935      0.437
                 RGBBQ        138        159      0.927      0.885      0.934      0.459
                 BMDBQ         61         71      0.776      0.634      0.698       0.26
                 BMGBQ        115        124      0.883      0.851      0.889      0.451
                   FBQ         14         17      0.893          1      0.989      0.618

yolov8l
                  all        509        604      0.902      0.886      0.904      0.479
                    BW         88         97      0.917      0.928      0.934      0.555
                    HQ        125        136       0.91      0.941      0.941      0.441
                 RGBBQ        138        159      0.947      0.903      0.947      0.487
                 BMDBQ         61         71      0.744      0.653      0.685       0.27
                 BMGBQ        115        124      0.932      0.891      0.921      0.492
                   FBQ         14         17      0.964          1      0.995      0.627

yolov8l 300 epoch
                  all        509        604      0.932      0.872       0.91      0.498
                    BW         88         97      0.948      0.938      0.926        0.6
                    HQ        125        136      0.941      0.919      0.937      0.481
                 RGBBQ        138        159      0.959      0.876      0.949      0.514
                 BMDBQ         61         71      0.803      0.662       0.75      0.293
                 BMGBQ        115        124      0.965      0.896       0.94      0.519
                   FBQ         14         17      0.978      0.941      0.958       0.58

yolo11l
                   all        509        604      0.915      0.869      0.913      0.478
                    BW         88         97      0.939      0.955      0.941      0.566
                    HQ        125        136      0.871      0.882      0.897      0.433
                 RGBBQ        138        159       0.94       0.89      0.954      0.499
                 BMDBQ         61         71      0.869      0.656      0.783      0.288
                 BMGBQ        115        124      0.941      0.887      0.923      0.485
                   FBQ         14         17      0.931      0.941       0.98      0.599


### CFG
YOLO11L
                   all        706        756      0.805      0.639      0.665      0.264      0.731      0.592      0.603      0.236
                    BW        183        194      0.821      0.861      0.855      0.335      0.782       0.83      0.823      0.294
                    HQ        270        286      0.756      0.749      0.714      0.242      0.617      0.619      0.565      0.178
                 RGBBQ        188        202      0.837      0.814      0.821       0.35      0.695      0.678       0.74       0.26
                 BMDBQ         20         23      0.746      0.512      0.605      0.164      0.416      0.304      0.396      0.136
                 BMGBQ         44         45      0.718      0.736      0.743      0.298      0.686      0.711      0.702      0.257
                    SC          1          1          1          0          0          0          1          0          0          0
                   FBQ          5          5      0.756        0.8       0.92       0.46      0.918          1      0.995      0.526
