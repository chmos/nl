/** Read file into a string */
static string file2str(const string& file) {
  ifstream t(file.c_str());
  stringstream os;
  os << t.rdbuf();
  t.close();
  return os.str();
}

/** Write a string to file */
static void str2file(const string& s, const string& file) {
  FILE *fp;
  errno_t et = fopen_s(&fp, file.c_str(), "w");
  if (fp != NULL) {
    fwrite(s.c_str(), 1, s.length(), fp);
    fclose(fp);
	}
}

/** split a string */
static void splitStr(const string& s, const string& dim,
                     std::vector<string>& seg) {
  seg.clear();
  
  if (dim.empty()) {
    seg.push_back(s);
    return;
	}

	size_t st = 0, ed = s.find(dim);
  while (ed != string::npos) {
    string a = (ed == st) ? "" : s.substr(st, ed - st);
    seg.push_back(a);

    st = ed + dim.length();
    ed = s.find(dim, st);
	}

	string b = s.substr(st);
  seg.push_back(b);
}


/** split and trim */
static void splitTrim(const string& s, const string& dim,
                      std::vector<string>& seg) {
  seg.clear();
  if (dim.empty()) {
    seg.push_back(Util::trim(s));
    return;
  }

  size_t st = 0, ed = s.find(dim);
  while (ed != string::npos) {
    string a = (ed == st) ? "" : s.substr(st, ed - st);
    seg.push_back(Util::trim(a));
    
    st = ed + dim.length();
    ed = s.find(dim, st);
	}

  string b = s.substr(st);
  seg.push_back(Util::trim(b));
}

/** trim the head and tail blanks */
static string trim(const string& s) {
  // string a = s;
  size_t st = s.find_first_not_of(" \f\n\r\t\v");
  size_t ed = s.find_last_not_of(" \f\n\r\t\v");
  return (st == string::npos || ed == string::npos) ? "" :
  s.substr(st, ed - st + 1);
}

/** remove extra whitespaces */
static string remove_extra_whitespaces(const string &input) {
  string output;
  // output.clear();  // unless you want to add at the end of existing sring...
  unique_copy(input.begin(), input.end(), back_insert_iterator<string>(output),
              [](char a, char b) { return isspace(a) && isspace(b); });
  // cout << output << endl;
  return output;
}

/** replace a substr */
static string replace(const string& s, const string& ol, const string& ne) {
  string a = s;
  size_t pos = a.find(ol);
  while (pos != string::npos) {
    a.replace(pos, ol.length(), ne);
    pos = a.find(ol);
  }

  return a;
}
