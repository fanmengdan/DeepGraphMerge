
## Requirements
- macOS/Windows/Linux
- Java JRE 8+
- MongoDB
- Git ^2.20.0

## Usage
### 1. Run MongoDB

### 2. Input Repo name
Enter the repository name in ***main*** of ***smartcommit\evaluation\ConstructGraph.java***, for example：
```sh
  String repoDir = "D:/javaRepo/"; 
  // name of the repo under test  
  String repoName = "glide"; 
  String repoPath = repoDir + repoName;
```

### 3. Input  the number of atomic commits 
Enter the expected number/step of atomic commits contained in each composite commit, in ***main*** of ***smartcommit\evaluation\ConstructGraph.java***,  for example:
```sh
// number of merged atomic commits to produce one composite commit  
int step = 2;
```

### 4. Set output path
Enter the expected output path, in ***main*** of ***smartcommit\evaluation\ConstructGraph.java***,  for example:
```sh
String outputPath = "D:/JavaProject/ConstructDependencyGraph/output/"+ repoName + step;
```
### 4. Run
Then run ***smartcommit\evaluation\ConstructGraph.java***, and the composite commits of **specific step** extracted from **the repository** will be obtained in the **output path**.

