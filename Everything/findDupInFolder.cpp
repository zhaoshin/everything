//
//  findDupInFolder.cpp
//  Everything
//
//  Created by Xing Zhao on 6/19/15.
//  Copyright (c) 2015 Zhao, Xing. All rights reserved.
//

#include <stdio.h>

static {
    try {
        md = MessageDigest.getInstance("SHA-512");
    } catch (NoSuchAlgorithmException e) {
        throw new RuntimeException("cannot initialize SHA-512 hash function", e);
    }
}

public static void find(Map<String, List<String>> lists, File directory, boolean leanAlgorithm) throws Exception  {
    String hash;
    for (File child : directory.listFiles()) {
        if (child.isDirectory()) {
            find(lists, child, leanAlgorithm);
        } else {
            try {
                hash = leanAlgorithm ? makeHashLean(child) : makeHashQuick(child);
                List<String> list = lists.get(hash);
                if (list == null) {
                    list = new LinkedList<String>();
                    lists.put(hash, list);
                }
                list.add(child.getAbsolutePath());
            } catch (IOException e) {
                throw new RuntimeException("cannot read file " + child.getAbsolutePath(), e);
            }
        }
    }
}

/*
 * quick but memory hungry (might like to run with java -Xmx2G or the like to increase heap space if RAM available)
 */
public static String makeHashQuick(File infile) throws Exception {
    FileInputStream fin = new FileInputStream(infile);
    byte data[] = new byte[(int) infile.length()];
    fin.read(data);
    fin.close();
    String hash = new BigInteger(1, md.digest(data)).toString(16);
    return hash;
}

#include <sys/stat.h>

size_t getFilesize(const char* filename) {
    struct stat st;
    if(stat(filename, &st) != 0) {
        return 0;
    }
    return st.st_size;
}