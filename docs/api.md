Here is the only method `gluon2pytorch` from `gluon2pytorch` module.

```
gluon2pytorch(net, args, dst_dir, pytorch_module_name, debug=True)
```

Options:

* `net` - a Gluon module to convert;
* `args` - list of shapes;
* `dst_dir` - None or path to destination directory;
* `pytorch_module_name` - string, a name of an output module;
* `debug` - debug output.