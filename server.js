const express = require('express');
const cors = require("cors");
const multer = require('multer');
const { PythonShell } = require('python-shell');
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const { applicationDefault } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');
const admin = require("firebase-admin");
const serviceAccount = require("/etc/secrets/firebase-admin-key.json"); // ← JSONファイルの相対パス
console.log("🔐 Admin SDK Project ID:", serviceAccount.project_id);
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});
const firestore = admin.firestore();
const app = express();
const port = 8080;

app.use(cors());
app.use(express.json());
app.use(express.static("public"));

// ✅ ルートパスにアクセスしたときのレスポンスを追加
app.get("/", (req, res) => {
  res.send("🎉 face-api-server is up and running!");
});

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const safeName = `${Date.now()}_${Math.random().toString(36).substring(2, 8)}${ext}`;
    cb(null, safeName);
  },
});

const upload = multer({ storage });

app.post('/analyze', upload.array('images', 5), (req, res) => {
  if (!req.files || req.files.length === 0) {
    console.error("❌ エラー: アップロードされた画像がありません");
    return res.status(400).json({ error: 'No images uploaded' });
  }

  const imagePaths = req.files.map(file => file.path);
  console.log("📸 ユーザーがアップロードした画像:", imagePaths);

  let options = {
    args: imagePaths,
    encoding: 'utf-8',
    pythonOptions: ['-Xutf8']
  };

  PythonShell.run('deepface_base.py', options)
    .then(messages => {
      console.log("📥 Python の出力:", messages);

      try {
        const jsonMessage = messages.find(msg => msg.trim().startsWith('['));
        if (!jsonMessage) throw new Error("Python の JSON 出力が見つかりません");

        const result = JSON.parse(jsonMessage);
        console.log("✅ JSON 解析成功:", result);
        res.json(result);

        // 🔹 成功時も確実に削除
        Promise.all(
          imagePaths.map(async (filePath) => {
            try {
              await fsp.unlink(filePath);
              console.log(`🧹 削除成功: ${filePath}`);
            } catch (err) {
              console.error(`⚠️ 削除失敗: ${filePath}`, err);
            }
          })
        );

      } catch (jsonError) {
        console.error("❌ JSON 解析エラー:", jsonError);
        res.status(500).json({ error: "Failed to parse response from Python script" });
      }
    })
    .catch(err => {
      console.error("❌ Python 実行エラー:", err);
      res.status(500).json({ error: "Internal Server Error", details: err.message });

      // 🔹 失敗時も削除
      Promise.all(
        imagePaths.map(async (filePath) => {
          try {
            await fsp.unlink(filePath);
            console.log(`🧹 (失敗時) 削除成功: ${filePath}`);
          } catch (err) {
            console.error(`⚠️ (失敗時) 削除失敗: ${filePath}`, err);
          }
        })
      );
    });
});

app.post("/save-latest", async (req, res) => {
  const result = req.body;

  try {
    // 各メンバーに対して Firestore から goodsLinks を補完
    for (const member of result) {
      const snap = await firestore.collection("members")
        .where("name", "==", member.name)
        .get();

      if (!snap.empty) {
        const data = snap.docs[0].data();
        member.goodsLinks = data.goodsLinks || null;
        member.profileUrl = data.profileUrl || null;
      } else {
        member.goodsLinks = null;
        member.profileUrl = null;
      }
    }

    // JSONファイルとして保存
    const filePath = path.join(__dirname, "public", "latest-result.json");
    fs.writeFile(filePath, JSON.stringify(result, null, 2), "utf-8", (err) => {
      if (err) {
        console.error("❌ 結果の保存に失敗:", err);
        return res.status(500).json({ error: "保存に失敗しました" });
      }
      console.log("✅ 診断結果（goodsLinks付）を保存しました:", filePath);
      res.json({ success: true });
    });
  } catch (error) {
    console.error("❌ Firestore 補完エラー:", error);
    res.status(500).json({ error: "Firestore補完エラー", details: error.message });
  }
});


app.get("/latest-result.json", (req, res) => {
  const filePath = path.join(__dirname, "public", "latest-result.json");
  fs.readFile(filePath, "utf-8", (err, data) => {
    if (err) {
      console.error("❌ 結果の読み込みに失敗:", err);
      return res.status(404).json({ error: "結果が見つかりません" });
    }
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.send(data);
  });
});

app.listen(port, () => {
  console.log(`🚀 Server running on http://localhost:${port}`);
});
