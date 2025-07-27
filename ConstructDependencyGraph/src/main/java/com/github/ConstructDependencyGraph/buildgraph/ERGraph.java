package com.github.ConstructDependencyGraph.buildgraph;
import com.github.ConstructDependencyGraph.core.GraphBuilder;
import com.github.ConstructDependencyGraph.model.DiffFile;
import com.github.ConstructDependencyGraph.model.graph.Edge;
import com.github.ConstructDependencyGraph.model.graph.Node;
import org.apache.commons.lang3.tuple.Pair;
import org.jgrapht.Graph;
import java.util.*;
import java.util.concurrent.*;


/** API entry */
public class ERGraph {

  // saved for analysis
  Graph<Node, Edge> baseGraph;
  Graph<Node, Edge> currentGraph;


  public ERGraph() {
    this.baseGraph = null;
    this.currentGraph = null;
  }

  /**
   * Build the Entity Reference Graphs for base and current versions
   *
   */
  private void buildRefGraphs(List<DiffFile> diffFiles, Pair<String, String> srcDirs)
          throws ExecutionException, InterruptedException, TimeoutException {
    ExecutorService executorService = Executors.newFixedThreadPool(2);
    Future<Graph<Node, Edge>> baseBuilder =
            executorService.submit(new GraphBuilder(srcDirs.getLeft(), diffFiles));
    Future<Graph<Node, Edge>> currentBuilder =
            executorService.submit(new GraphBuilder(srcDirs.getRight(), diffFiles));
    baseGraph = baseBuilder.get(60 * 10, TimeUnit.SECONDS);
    currentGraph = currentBuilder.get(60 * 10, TimeUnit.SECONDS);
    executorService.shutdown();
  }

  public Graph<Node, Edge> getBaseGraph() {
    return baseGraph;
  }


  /**
   * buildRefGraphs
   *
   */
  public void analyze(List<DiffFile> diffFiles, Pair<String, String> srcDirs) {
    try {
      buildRefGraphs(diffFiles, srcDirs);
    } catch (Exception e) {
      System.err.println("Exception during graph building:");
      e.printStackTrace();
    }
  }

}
