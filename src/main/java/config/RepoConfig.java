package config;

import java.util.List;
import java.util.Map;

public interface RepoConfig
{
    String getRepo();

    String getBranch();

    List<String> getFiles();

    Map<String, String> getFileNameOverrides();
}
