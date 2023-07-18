import 'package:flutter/material.dart';
import 'package:app/services/model_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FitChat',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(title: 'FitChat'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final TextEditingController _controller = TextEditingController();
  String? _userInput;
  String? _pred;

  void _receiveInput() {
    setState(() {
      _userInput = _controller.text;
    });
    _controller.clear();
  }

  Future<String> _getPrediction() async {
    ModelService ms = ModelService();
    if (_userInput == null) {
      _pred = "Welcome to FitChat! Type something to get started.";
    } else {
      _pred = await ms.getPred(_userInput!);
    }
    print(_pred!);
    return _pred!;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: const Text('FitChat'),
          centerTitle: true,
        ),
        body: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                TextField(
                  controller: _controller,
                  decoration: InputDecoration(
                      hintText: "Please ask anything fitness-related!",
                      border: const OutlineInputBorder(),
                      suffixIcon: IconButton(
                          onPressed: _receiveInput,
                          icon: const Icon(Icons.send))),
                ),
                FutureBuilder<String>(
                    builder:
                        (BuildContext context, AsyncSnapshot<String> snapshot) {
                      if (snapshot.connectionState == ConnectionState.done) {
                        if (snapshot.hasError) {
                          return Center(
                            child: Text(
                              '${snapshot.error} occurred',
                              style: TextStyle(fontSize: 18),
                            ),
                          );
                        } else if (snapshot.hasData) {
                          var data = snapshot.data as String;
                          return Container(
                              alignment: Alignment.topCenter,
                              width: 500,
                              height: 400,
                              decoration: BoxDecoration(
                                border: Border.all(
                                  color: Colors.grey,
                                  width: 1.0,
                                ),
                                borderRadius: BorderRadius.circular(4.0),
                              ),
                              child: ListView(
                                children: <Widget>[
                                  Padding(
                                    padding: EdgeInsets.all(8.0),
                                    child: Text(
                                      data,
                                      style: TextStyle(fontSize: 16),
                                    ),
                                  )
                                ],
                              ));
                        }
                      }
                      return Center(
                        child: CircularProgressIndicator(),
                      );
                    },
                    future: _getPrediction()),
              ]),
        ));
  }
}
