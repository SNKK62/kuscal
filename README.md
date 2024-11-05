# kuscal

This is the toy language inspired by ruscal ([Original Source](https://github.com/msakuta/ruscal)).<br>
I implemented this with reading 「Creating Programming Languages in Rust」, and customized it for my training.<br>
In this language, `.kscl` code is compiled to bytecode and executed by the virtual machine.

## Original Features

### while statement
#### source
```
var a: f64 = 0;

while (a < 10)  {
    println(a);
    a = a + 1;
}
```

#### output
```
0 1 2 3 4 5 6 7 8 9
```

### comparison operator
Basic comparison operators are available. (`<`, `>`, `==`, `!=`).<br>
`>=` and `<=` are not available now, but they can be implemented easily.

### not operator
`!` operator is available.

### Array
Array is available. You can access elements by index.<br>
Multi-dimensional array is also available!

### source
```
var arr: Array<Array<f64>[8]>[8] = [];

arr[3][3] = 1;
arr[3][4] = 2;
arr[4][3] = 3;
arr[4][4] = 4;

for i in 0 to 8 {
    for j in 0 to 8 {
        print(arr[i][j], " ");
    }
    println("");
}
```

### output
```
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 1 2 0 0 0
0 0 0 3 4 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
```

## Type
Type check is executed at compile time.<br>
Now, only `f64`, `str`, `Array<Type>[Size]` are available.<br>
`i64` is used internally, but it is not available in the language.

## example code
### Bubble Sort
#### source
```
var arr: Array<f64>[10] = [2, 4, 6, 8, 10, 9, 7, 5, 3, 1];

println("[Original]");
for k in 0 to 10 {
    print(arr[k], " ");
}
println("");
println("");

for i in 0 to 10 {
    var is_changed: f64 = 0;
    for j in 0 to (10 - 1 - i) {
        if arr[j] > arr[j + 1] {
            var temp: f64 = arr[j];
            arr[j] = arr[j + 1];
            arr[j + 1] = temp;
            is_changed = 1;
        };
    }

    print("[", i, "]: ");
    for k in 0 to 10 {
        print(arr[k], " ");
    }
    println("");
    if !is_changed {
        break;
    }
}

println("");
println("[Answer]");
for k in 0 to 10 {
    print(arr[k], " ");
}
println("");
```

#### output
```
[Original]
2 4 6 8 10 9 7 5 3 1

[0]: 2 4 6 8 9 7 5 3 1 10
[1]: 2 4 6 8 7 5 3 1 9 10
[2]: 2 4 6 7 5 3 1 8 9 10
[3]: 2 4 6 5 3 1 7 8 9 10
[4]: 2 4 5 3 1 6 7 8 9 10
[5]: 2 4 3 1 5 6 7 8 9 10
[6]: 2 3 1 4 5 6 7 8 9 10
[7]: 2 1 3 4 5 6 7 8 9 10
[8]: 1 2 3 4 5 6 7 8 9 10
[9]: 1 2 3 4 5 6 7 8 9 10

[Answer]
1 2 3 4 5 6 7 8 9 10
```


