const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, AlignmentType, WidthType, BorderStyle, VerticalAlign, LineRuleType } = require('docx');

const FONT = 'Times New Roman', SZ = 24; // 12 pt

// Segmen di antara tanda * dicetak miring.
function runs(text, base = {}) {
  return text.split('*').map((seg, i) => new TextRun({ text: seg, font: FONT, size: SZ, ...base, italics: i % 2 === 1 }));
}
const body = (text) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 720 },
  spacing: { line: 480, lineRule: LineRuleType.AUTO, after: 0 }, children: runs(text) });
const caption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 }, keepNext: true, children: runs(text) });
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });

const B = { style: BorderStyle.SINGLE, size: 6, color: '000000' };
const borders = { top: B, bottom: B, left: B, right: B };
const C = AlignmentType.CENTER, L = AlignmentType.LEFT;

function table(widths, header, rows, aligns) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, i, head) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: head ? C : (aligns[i] || C),
      children: runs(t, { size: 22, bold: head }) })] });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, widths[i], i, true)) }),
      ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], i, false)) }))] });
}

const children = [
  caption('Tabel 4.1 Dataset *Modelling* Kuku'),
  table([2100, 2800, 4100], ['Dataset', 'File Utama', 'Isi'], [
    ['*Anemia-survey*', 'fingernails_open/ (.jpeg)', 'Citra kuku tiga jari pada posisi tangan terbuka'],
    ['*Anemia-survey*', 'fingernails_closed/ (.jpeg)', 'Citra kuku tiga jari pada posisi tangan mengepal'],
    ['*Anemia-survey*', 'metadata.csv', 'ID pasien, Hb laboratorium (g/dL), kategori anemia, usia, jenis kelamin, status kehamilan, dan jenis perangkat'],
    ['*MSU-250* (eksternal)', '<ID pasien>.jpg', 'Citra kuku pasien yang digunakan hanya untuk pengujian'],
    ['*MSU-250* (eksternal)', 'metadata.csv', 'ID pasien, tanggal pengukuran, dan Hb laboratorium (g/L)'],
    ['*NailSegmentationDatasetV2*', 'images/ (.jpg), masks/ (.png)', 'Citra tangan/kuku dan *mask* segmentasi kuku untuk melatih model segmentasi'],
  ], [C, C, L]),
  gap(),
  body('Berdasarkan Tabel 4.1, dataset *Anemia-survey* bersumber dari repositori Hugging Face *sewa-rural-care/anemia-survey-dataset* yang dikumpulkan oleh SEWA Rural, organisasi nirlaba layanan kesehatan primer yang melayani komunitas pedesaan dan suku di Gujarat, India, sejak tahun 1968. Dataset ini mendukung penelitian skrining anemia non-invasif berbasis *smartphone* pada layanan kesehatan dengan sumber daya terbatas, serta menyediakan citra kuku dalam dua posisi tangan, yaitu terbuka (*open*) dan mengepal (*closed*), beserta file *metadata* yang memuat nilai Hb hasil pemeriksaan laboratorium dan data demografi pasien. Dataset *MSU-250* bersumber dari *Dataset of human skin and fingernails images for non-invasive haemoglobin level assessment* yang dipublikasikan pada *Scientific Data* (2024) oleh kelompok *biophotonics* MSU dan tersedia secara terbuka melalui repositori GitHub *biophotonics-msu/photo-haemoglobin* dengan lisensi MIT. Dataset ini terdiri atas satu citra kuku untuk setiap pasien dan nilai Hb dalam satuan g/L; dataset ini tidak pernah digunakan pada proses pelatihan maupun validasi silang, sehingga berfungsi sebagai data uji eksternal. Dataset *NailSegmentationDatasetV2* diperoleh dalam bentuk ekspor Roboflow [LENGKAPI: nama pembuat dan tautan sumber], tidak memiliki nilai Hb, dan hanya digunakan untuk melatih model segmentasi kuku.'),

  caption('Tabel 4.2 Jumlah Dataset *Modelling* Kuku'),
  table([2300, 1300, 1400, 1500, 1500, 1000], ['Dataset', 'Subjek', 'Citra *Open*', 'Citra *Closed*', 'Total File Citra', 'Satuan Hb'], [
    ['*Anemia-survey*', '5.782', '5.782', '5.782', '11.564', 'g/dL'],
    ['*MSU-250*', '250', '250', '–', '250', 'g/L'],
    ['*NailSegmentationDatasetV2*', '–', '–', '–', '7.012', '–'],
  ], [C, C, C, C, C, C]),
  gap(),
  body('Berdasarkan Tabel 4.2, dataset *Anemia-survey* terdiri atas 5.782 subjek yang masing-masing memiliki satu citra kuku terbuka dan satu citra kuku mengepal, sehingga totalnya 11.564 citra. File *metadata* memuat 5.869 baris karena 87 subjek tidak memiliki citra kuku (hanya citra bagian tubuh lain yang di luar cakupan penelitian ini); seluruh baris tersebut memiliki nilai Hb. Dataset *MSU-250* terdiri atas 250 subjek dengan satu citra per subjek. Nilai Hb pada kedua dataset memiliki satuan yang berbeda (g/dL dan g/L) sehingga disamakan terlebih dahulu sebelum dievaluasi. Dataset *NailSegmentationDatasetV2* tidak berbasis pasien dan tidak memiliki nilai Hb; dataset ini terdiri atas 7.012 citra beserta 7.012 *mask* segmentasi kuku, dengan rincian pembagian data latih, validasi, dan uji pada Tabel 4.3.'),

  caption('Tabel 4.3 Jumlah Dataset Segmentasi Kuku'),
  table([2800, 2000, 2000, 2200], ['Pembagian Data', 'Citra', '*Mask*', 'Total File'], [
    ['*Train*', '5.620', '5.620', '11.240'],
    ['*Validation*', '690', '690', '1.380'],
    ['*Test*', '702', '702', '1.404'],
    ['Total', '7.012', '7.012', '14.024'],
  ], [C, C, C, C]),
  gap(),
  body('Berdasarkan Tabel 4.3, dataset segmentasi terbagi menjadi data latih, validasi, dan uji dengan jumlah citra yang sama banyak dengan jumlah *mask*-nya. Dataset ini digunakan untuk melatih YOLO26-seg agar dapat menghasilkan *mask* pada setiap kuku sebelum area kuku dipotong dan diproses oleh model estimasi Hb.'),

  caption('Tabel 4.4 Distribusi Kategori Anemia pada Dataset *Anemia-survey*'),
  table([3200, 2000, 2000], ['Kategori', 'Jumlah Subjek', 'Persentase'], [
    ['Normal', '953', '16,2%'],
    ['Anemia ringan (*mild*)', '1.325', '22,6%'],
    ['Anemia sedang (*moderate*)', '2.680', '45,7%'],
    ['Anemia berat (*severe*)', '911', '15,5%'],
    ['Total', '5.869', '100,0%'],
  ], [L, C, C]),
  gap(),
  body('Berdasarkan Tabel 4.4, sebanyak 4.916 subjek (83,8%) tergolong anemia dan hanya 953 subjek (16,2%) yang normal. Nilai Hb pada dataset ini berkisar antara 1,3 hingga 19,2 g/dL dengan rata-rata 10,09 g/dL. Sebaran yang condong ke anemia ini perlu diperhatikan, karena model cenderung memprediksi nilai Hb mendekati rata-rata sehingga kesalahan lebih besar pada Hb yang sangat rendah maupun tinggi.'),

  caption('Tabel 4.5 Data yang Digunakan Setelah Penyaringan'),
  table([2900, 1500, 1800, 2800], ['Dataset', 'Subjek', '*Crop* Kuku', 'Peran'], [
    ['*Anemia-survey*', '5.257', '47.995', 'Pelatihan dan validasi silang 5-*fold*'],
    ['*MSU-250*', '250', '1.059', 'Pengujian eksternal'],
  ], [C, C, C, C]),
  gap(),
  body('Berdasarkan Tabel 4.5, dari 5.782 subjek pada dataset *Anemia-survey* terdapat 5.257 subjek yang digunakan setelah penyaringan, yaitu foto dengan latar terang, kuku yang berhasil terdeteksi oleh model segmentasi, dan nilai Hb pada rentang 3–20 g/dL. Dari subjek tersebut diperoleh 47.995 *crop* kuku dengan batas maksimal enam kuku untuk setiap posisi tangan. Pembagian data pada validasi silang dilakukan per pasien sehingga seluruh *crop* dari satu pasien selalu berada pada *fold* yang sama.'),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1701, bottom: 1701, left: 1701, right: 1134 } } }, children }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync('laporan_dataset_kuku_v3.docx', b); console.log('ok'); });
