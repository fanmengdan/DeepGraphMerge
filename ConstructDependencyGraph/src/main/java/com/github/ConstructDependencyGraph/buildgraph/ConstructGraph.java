package com.github.ConstructDependencyGraph.buildgraph;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.revwalk.RevCommit;
import com.github.ConstructDependencyGraph.core.RepoAnalyzer;
import com.github.ConstructDependencyGraph.io.DataCollector;
import com.github.ConstructDependencyGraph.model.*;
import com.github.ConstructDependencyGraph.model.constant.FileType;
import com.github.ConstructDependencyGraph.model.constant.Version;
import com.github.ConstructDependencyGraph.model.graph.Edge;
import com.github.ConstructDependencyGraph.model.graph.Node;
import com.github.ConstructDependencyGraph.model.graph.NodeType;
import com.github.ConstructDependencyGraph.util.Utils;
import com.mongodb.MongoClient;
import com.mongodb.MongoClientURI;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import com.mongodb.client.MongoDatabase;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.bson.Document;
import org.jgrapht.Graph;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public class ConstructGraph {
    private static final String dataDir = "D:/JavaProject/ConstructDependencyGraph/tempDir/";
    private static final String mongoDBUrl = "mongodb://localhost:27017";

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();
        org.apache.log4j.Logger.getRootLogger().setLevel(Level.INFO);

        String repoDir = "D:/javaRepo/";
        String tempDir = dataDir;

        String repoName = "druid";
        int step = 2;
        String repoPath = repoDir + repoName;

        initMongoData(repoPath, repoName);

        runOpenSrc(repoName, repoPath, tempDir + "/" + repoName, step);
    }

    public static void initMongoData(String repoPath, String repoName) {
        MongoClient mongoClient = new MongoClient(new MongoClientURI(mongoDBUrl));
        MongoCollection<Document> col = mongoClient.getDatabase("atomic").getCollection(repoName);

        try (Git git = Git.open(new File(repoPath))) {
            Iterable<RevCommit> commits = git.log().call();
            for (RevCommit commit : commits) {
                Document doc = new Document()
                        .append("commit_id", commit.getName())
                        .append("committer_email", commit.getCommitterIdent().getEmailAddress());
                col.insertOne(doc);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        mongoClient.close();
    }

    private static void runOpenSrc(String repoName, String repoPath, String outputDir, int step) throws Exception {
        System.out.println("Open Source Repo: " + repoName + " Step: " + step);

        String tempDir = outputDir + "/" + step;
        Utils.clearDir(tempDir);

        MongoClientURI connectionString = new MongoClientURI(mongoDBUrl);
        MongoClient mongoClient = new MongoClient(connectionString);
        MongoDatabase db = mongoClient.getDatabase("atomic");
        MongoCollection<Document> col = db.getCollection(repoName);

        Map<String, List<String>> commitsByEmail = new HashMap<>();
        try (MongoCursor<Document> cursor = col.find().iterator()) {
            while (cursor.hasNext()) {
                Document doc = cursor.next();
                String email = (String) doc.get("committer_email");
                String commitID = (String) doc.get("commit_id");
                if (commitsByEmail.containsKey(email)) {
                    commitsByEmail.get(email).add(commitID);
                } else {
                    List<String> ids = new ArrayList<>();
                    ids.add(commitID);
                    commitsByEmail.put(email, ids);
                }
            }
        }
        mongoClient.close();

        Map<String, List<String>> commitsByEmailAboveStep =
                commitsByEmail.entrySet().stream()
                        .filter(a -> a.getValue().size() >= step)
                        .collect(Collectors.toMap(Entry::getKey, Entry::getValue));

        ERGraph ERGraph =
                new ERGraph();

        int sampleNum = 0;
        for (Entry<String, List<String>> entry : commitsByEmailAboveStep.entrySet()) {
            List<String> commits = entry.getValue();

            outerloop:
            for (int i = 0; i < commits.size(); i += 1) {
                if (sampleNum >= 100) {
                    break outerloop;
                }
                Map<String, Set<String>> groundTruth = new LinkedHashMap<>();
                List<DiffFile> unionDiffFiles = new ArrayList<>();
                List<DiffHunk> unionDiffHunks = new ArrayList<>();
                Map<String, DiffHunk> unionDiffHunkMap = new HashMap<>();

                String resultsDir = tempDir + File.separator + commits.get(i);

                DataCollector dataCollector = new DataCollector(repoName, resultsDir);

                int LOC = 0;
                int j;
                String dirNameHash = "";
                for (j = 0; j < step; j++) {
                    if (i + j < commits.size()) {
                        String commitID = commits.get(i + j);
                        if (!commitID.isEmpty()) {
                            if (dirNameHash.isEmpty()) {
                                dirNameHash = commitID.substring(0, 7);
                            } else {
                                dirNameHash = dirNameHash + "_" + commitID.substring(0, 7);
                            }

                            RepoAnalyzer repoAnalyzer =
                                    new RepoAnalyzer(String.valueOf(repoName.hashCode()), repoName, repoPath);

                            List<DiffFile> diffFiles = repoAnalyzer.analyzeCommit(commitID);

                            dataCollector.collectDiffFilesWorking(diffFiles);

                            int beginIndex = unionDiffFiles.size();

                            for (int k = 0; k < diffFiles.size(); ++k) {
                                int newIndex = beginIndex + k;
                                diffFiles.get(k).setIndex(Integer.valueOf(newIndex));
                                for (DiffHunk diffHunk : diffFiles.get(k).getDiffHunks()) {
                                    diffHunk.setFileIndex(Integer.valueOf(newIndex));
                                    LOC +=
                                            (diffHunk.getBaseEndLine() - diffHunk.getBaseStartLine() + 1)
                                                    + (diffHunk.getCurrentEndLine() - diffHunk.getCurrentStartLine() + 1);
                                }
                            }
                            List<DiffHunk> diffHunks = repoAnalyzer.getDiffHunks();
                            if (diffHunks.isEmpty()) {
                                continue;
                            }

                            unionDiffFiles.addAll(diffFiles);
                            unionDiffHunks.addAll(diffHunks);

                            unionDiffHunkMap.putAll(repoAnalyzer.getIdToDiffHunkMap());

                            groundTruth.put(commitID, repoAnalyzer.getIdToDiffHunkMap().keySet());
                        }
                    } else {
                        break;
                    }
                }

                if (j < step || unionDiffHunks.size() > 200 || checkPureNoJavaChanges(unionDiffHunks)) {
                    FileUtils.deleteQuietly(new File(resultsDir));
                    continue;
                }

                System.out.println("____________________________________");
                System.out.println("[Batch " + sampleNum + ":" + dirNameHash + "] groundtruth:");
                for (Map.Entry<String, Set<String>> s : groundTruth.entrySet()) {
                    System.out.println(s);
                }
                System.out.println();

                String outputPath = "D:/JavaProject/ConstructDependencyGraph/output/"+ repoName + "/" + step;
                File f1 = new File(outputPath);
                if(!f1.exists()){
                    f1.mkdirs();
                }

                String baseDir = resultsDir + File.separator + Version.BASE.asString() + File.separator;
                String currentDir =
                        resultsDir + File.separator + Version.CURRENT.asString() + File.separator;

                ERGraph.analyze(unionDiffFiles, Pair.of(baseDir, currentDir));

                Graph<Node, Edge> baseGraph = ERGraph.getBaseGraph();

                List<Integer> IdList = new ArrayList<>();
                List<String> DiffIndexList = new ArrayList<>();
                List<NodeType> NodeTypeList = new ArrayList<>();

                for (Node node : baseGraph.vertexSet()){
                    IdList.add(node.getId());
                    DiffIndexList.add(node.getdiffhunkID());
                    NodeTypeList.add(node.getType());
                }

                String Graphpath = outputPath + "/" + dirNameHash;
                File f = new File(Graphpath);
                if(!f.exists()){
                    f.mkdirs();
                }

                String savepath = Graphpath + "/GT_" +  dirNameHash + ".txt";
                saveMapToTxt(savepath, groundTruth);

                String IdPath = Graphpath + "/Id_" +  dirNameHash + ".txt";
                String DiffIndexPath = Graphpath + "/Index_" + dirNameHash + ".txt";
                String NodeTypePath = Graphpath + "/Type_" + dirNameHash + ".txt";
                writeIntList(IdPath,IdList);
                writeStringList(DiffIndexPath,DiffIndexList);
                writeNodeTypeList(NodeTypePath,NodeTypeList);

                Set<Edge> baseEdgeset = baseGraph.edgeSet();
                List<Integer> sourceIdList = new ArrayList<>();
                List<Integer> targetIdList = new ArrayList<>();

                for (Edge e : baseEdgeset){
                    sourceIdList.add(baseGraph.getEdgeSource(e).getId());
                    targetIdList.add(baseGraph.getEdgeTarget(e).getId());
                }
                System.out.println("sourceId size:" + sourceIdList.size());
                System.out.println("targetId size:" + targetIdList.size());

                String SourceIdPath = Graphpath + "/Source_" + dirNameHash + ".txt";
                String TargetIdPath = Graphpath + "/Target_" + dirNameHash + ".txt";
                writeIntList(SourceIdPath,sourceIdList);
                writeIntList(TargetIdPath,targetIdList);

                sampleNum++;
            }
        }
    }

    static void writeIntList (String path, List<Integer> listName) throws IOException {
        BufferedWriter bufferedWriter = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));

        for (Integer node : listName) {
            bufferedWriter.write(node.toString());
            bufferedWriter.newLine();
            bufferedWriter.flush();
        }
        bufferedWriter.close();
    }

    static void writeStringList (String path, List<String> listName) throws IOException {
        BufferedWriter bufferedWriter = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));

        for (String index : listName) {
            bufferedWriter.write(index.toString());
            bufferedWriter.newLine();
            bufferedWriter.flush();
        }
        bufferedWriter.close();
    }

    static void writeNodeTypeList (String path, List<NodeType> listName) throws IOException {
        BufferedWriter bufferedWriter = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));

        for (NodeType index : listName) {
            bufferedWriter.write(index.toString());
            bufferedWriter.newLine();
            bufferedWriter.flush();
        }
        bufferedWriter.close();
    }

    private static boolean checkPureNoJavaChanges(List<DiffHunk> diffHunks) {
        Optional<DiffHunk> javaDiffHunk =
                diffHunks.stream()
                        .filter(diffHunk -> diffHunk.getFileType().equals(FileType.JAVA))
                        .findAny();
        return !javaDiffHunk.isPresent();
    }

    public static void saveMapToTxt(String filepath, Map<String, Set<String>> map) {
        try {
            String line = System.getProperty("line.separator");
            StringBuffer str = new StringBuffer();
            PrintWriter pw = new PrintWriter(filepath);

            Set set = map.entrySet();
            Iterator iter = set.iterator();
            while(iter.hasNext()){
                Map.Entry entry = (Map.Entry)iter.next();
                str.append(entry.getKey()+" : "+entry.getValue()).append(line);
            }

            pw.write(str.toString());
            pw.flush();
            pw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}