
val sparkVersion = "2.3.0"

lazy val sparkSettings = Seq(
    organization := "org.freedom",
    version := "0.1",
    scalaVersion := "2.11.12",
    libraryDependencies := Seq(
        "org.scala-lang" % "scala-reflect" % scalaVersion.value % "provided",
        "org.apache.spark" %% "spark-mllib" % sparkVersion ,
        "org.apache.spark" % "spark-sql_2.11" % sparkVersion % "provided",
        "org.apache.spark" % "spark-streaming_2.11" % sparkVersion % "provided",
        "org.apache.spark" % "spark-streaming-kafka-0-10_2.11" % sparkVersion % "provided"
    )
)

lazy val root = (project in file("."))
        .settings(
            sparkSettings,
            name := "spark-tool",
            //            unmanagedBase := baseDirectory.value / "src/main/resources",
            mainClass in assembly := Some("Main"),
            assemblyJarName in assembly := "test-assembly.jar",
            assemblyMergeStrategy in assembly := {
                case PathList("javax", "servlet", xs @ _*) => MergeStrategy.last
                case PathList("javax", "inject", xs @ _*) => MergeStrategy.last
                case PathList("javax", "activation", xs @ _*) => MergeStrategy.last
                case PathList("org", "apache", xs @ _*) => MergeStrategy.last
                case PathList("org", "aopalliance", xs @ _*) => MergeStrategy.last
                case PathList("net", "jpountz", xs @ _*) => MergeStrategy.last
                case PathList("com", "google", xs @ _*) => MergeStrategy.last
                case PathList("com", "esotericsoftware", xs @ _*) => MergeStrategy.last
                case PathList("com", "codahale", xs @ _*) => MergeStrategy.last
                case PathList("com", "yammer", xs @ _*) => MergeStrategy.last
                case "about.html" => MergeStrategy.rename
                case "META-INF/mailcap" => MergeStrategy.last
                case "META-INF/mimetypes.default" => MergeStrategy.last
                case "plugin.properties" => MergeStrategy.last
                case "git.properties" => MergeStrategy.last
                case "log4j.properties" => MergeStrategy.last
                case "overview.html" => MergeStrategy.filterDistinctLines
                case x =>
                    val oldStrategy = (assemblyMergeStrategy in assembly).value
                    oldStrategy(x)
            },
            assemblyShadeRules in assembly := Seq(
            ),
            assemblyExcludedJars in assembly := {
                val jars = Seq(
                    "c3p0-0.9.5.2.jar",
                    "config-1.2.1.jar",
                    "hadoop-lzo-0.4.20.jar",
                    "imlib_2.11-0.0.1.jar",
                    "jedis-2.1.0.jar",
                    "mariadb-java-client-1.5.9.jar",
                    "mchange-commons-java-0.2.11.jar",
                    "mysql-connector-java-5.1.41-bin.jar",
                    "protobuf-java-3.0.2.jar",
                    "scala-redis_2.11-1.0.jar",
                    "spark-redis-0.3.2.jar",
                    "spark-streaming-kafka-0-8-assembly_2.11-2.1.0.jar",
                    "xgboost4j-0.7.jar",
                    "xgboost4j-spark-0.7.jar",
                    "scala-library-2.11.8.jar"
                )
                (fullClasspath in assembly).value.filter {
                    x => jars.contains(x.data.getName)
                }
            }
        )


