# Matlab Interface packaging functions

Simply run `package_osqp.m`. It will compile the interface, package it and upload it to the GitHub release.

You need:

-   The Github repository token (ask the developers).
-   The [github-release](https://github.com/aktau/github-release) Go language package installed and its executable added to the Matlab path. To do so, simply add on Unix systems

```matlab
setenv('PATH', [getenv('PATH') ':$GOPATH/bin']);
```
where `$GOPATH` is the path to the go installation. On Windows system add

```matlab
setenv('PATH', [getenv('PATH') ';%GOPATH%/bin']);
```

Note the difference of `:` and `;` between Unix and Windows platforms.
