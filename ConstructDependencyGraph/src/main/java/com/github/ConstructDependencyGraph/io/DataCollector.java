package com.github.ConstructDependencyGraph.io;
import com.github.ConstructDependencyGraph.model.DiffFile;
import com.github.ConstructDependencyGraph.model.constant.*;
import com.github.ConstructDependencyGraph.util.Utils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import java.io.File;
import java.util.*;


public class DataCollector {
  private static final Logger logger = Logger.getLogger(DataCollector.class);

  private String repoName;
  private String tempDir;

  public DataCollector(String repoName, String tempDir) {
    this.repoName = repoName;
    this.tempDir = Utils.createDir(tempDir);
  }


  /**
   * Collect the base and current version of diff files
   *
   * @return
   */
  public Pair<String, String> collectDiffFilesWorking(List<DiffFile> diffFiles) {
    String baseDir = tempDir + File.separator + Version.BASE.asString() + File.separator;
    String currentDir = tempDir + File.separator + Version.CURRENT.asString() + File.separator;

    collect(baseDir, currentDir, diffFiles);
    return Pair.of(baseDir, currentDir);
  }

  /**
   * Collect the diff files into the data dir
   *
   * @param baseDir
   * @param currentDir
   * @param diffFiles
   * @return
   */
  private int collect(String baseDir, String currentDir, List<DiffFile> diffFiles) {
    int count = 0;
    Utils.createDir(baseDir);
    Utils.createDir(currentDir);
    for (DiffFile diffFile : diffFiles) {
      // skip binary files
      if (diffFile.getFileType().equals(FileType.BIN)) {
        continue;
      }
      String basePath, currentPath;
      switch (diffFile.getStatus()) {
        case ADDED:
        case UNTRACKED:
          currentPath = currentDir + diffFile.getCurrentRelativePath();
          if (Utils.writeStringToFile(diffFile.getCurrentContent(), currentPath)) {
            count++;
          } else {
            logger.error("Error when collecting: " + diffFile.getStatus() + ":" + currentPath);
          }
          break;
        case DELETED:
          basePath = baseDir + diffFile.getBaseRelativePath();
          if (Utils.writeStringToFile(diffFile.getBaseContent(), basePath)) {
            count++;
          } else {
            logger.error("Error when collecting: " + diffFile.getStatus() + ":" + basePath);
          }
          break;
        case MODIFIED:
        case RENAMED:
        case COPIED:
          basePath = baseDir + diffFile.getBaseRelativePath();
          currentPath = currentDir + diffFile.getCurrentRelativePath();
          boolean baseOk = Utils.writeStringToFile(diffFile.getBaseContent(), basePath);
          boolean currentOk = Utils.writeStringToFile(diffFile.getCurrentContent(), currentPath);
          if (baseOk && currentOk) {
            count++;
          } else {
            logger.error("Error when collecting: " + diffFile.getStatus() + ":" + basePath);
          }
          break;
      }
    }
    return count;
  }

}
