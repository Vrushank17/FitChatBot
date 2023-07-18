import 'package:http/http.dart' as http;
import 'dart:convert';

class ModelService {
  Future<String> getPred(String userInput) async {
    var funcUrl = "http://localhost:5000/get_response/" + userInput;
    var uri = Uri.parse(funcUrl);
    var response = await http.get(uri);

    if (response.statusCode == 200) {
      var jsonResponse = json.decode(response.body);
      return jsonResponse['value'];
    } else {
      throw Exception("Invalid prediction");
    }
  }
}
