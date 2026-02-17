# Transformer Mimarisini Anlamak

![resim.png](Transformer Mimarisini Anlamak_files/13fe1473-c716-4bf5-b8d8-23e27fccbf46.png)

Yinelemeli sinir ağları (RNN), özellikle **Long Short-Term Memory (LSTM)** ve **Gated Recurrent Unit (GRU)** modelleri, dil modelleme ve makine çevirisi gibi sıralı veri problemlerinde en iyi yöntemler olarak yerleşmiştir. Bu alanda birçok çalışma, RNN tabanlı dil modellerini ve encoder-decoder mimarilerini daha da geliştirmeye devam etmiştir.
RNN modelleri genellikle hesaplamayı giriş ve çıkış dizisinin sembol pozisyonlarına göre böler. Her zaman adımında (t), önceki gizli durum (hₜ₋₁) ve o pozisyondaki giriş kullanılarak yeni bir gizli durum (hₜ) üretilir.
Bu yapı doğası gereği sıralıdır (sequential). Yani zaman adımları paralel çalıştırılamaz. Bu durum özellikle uzun dizilerde ciddi bir problem olur çünkü:

* Paralelleştirme yapılamaz
* Bellek kısıtları nedeniyle batch boyutu sınırlanır

Bazı çalışmalar hesaplama verimliliğini artırmak için faktörizasyon teknikleri ve koşullu hesaplama yöntemleri kullanmıştır. Ancak temel sorun olan **sıralı hesaplama zorunluluğu** devam etmektedir.

Attention mekanizmaları, giriş veya çıkış dizisindeki uzak bağımlılıkları modelleyebilmek için geliştirilmiştir. Mesafe önemli olmadan bağımlılıkları öğrenebilirler.

## 1. Adım Verisetini Tanımlama

Elimizde şöyle bir veri seti oldugunu varsayalım: 

![1.png](Transformer Mimarisini Anlamak_files/2c007de7-641f-43b1-b2a5-d955afb1b4f8.png)

Verisetimizi 3 cümleden oluştuğunu görebiliyoruz. ChatGPT gibi modellerin yüzlerce GB veriden oluştuğunu unutmamak gerekir.

## 2. Adım Verisetimizdeki eşsiz sayıdaki kelimeleri bularak sözlük oluşturmak
Sözlük boyutu toplem eşsiz kelime sayısını tanımlar formül olarka:

$vocab size = count (set(N))$

N değişkeni burada toplam kelime sayısına eşittir

N değerini bulabilmek için verisetini kelimelere bölmek gerekir

![resim.png](Transformer Mimarisini Anlamak_files/17e9d2c7-0eb7-4f3b-8c74-1e0f3e8af40b.png)

N  değerini bulduktan sonra tekrarlayan kelimleri temizleyerek gerçek sözlük boyutunu bulmamız gerekmektedir.

![resim.png](Transformer Mimarisini Anlamak_files/6bbdd742-b516-4675-a4f2-53c72222f9a4.png)


## 3. Adım Encoding (Kodlama)
Şimdi bu sözlükteki her eşsiz kelimeye karşılık gelen bir rakam vermemiz gerekmekte.
![resim.png](Transformer Mimarisini Anlamak_files/d0fc2364-b248-4c20-b429-cba3f58d71bc.png)

Tüm verisetini kodladıktan sonra şimdi girdi verisi seçerek transformer mimarisi üzerinde çalışabiliriz


## 4. Adım Kodlamayı hesaplamak
Şİmdi bir mevcut corpus'umuz içeriğinden bir cümle seçelim ve o cümleye göre transformers mimarisini inceleyelim

Cümle:
"When you play game of thrones"
![resim.png](Transformer Mimarisini Anlamak_files/78c9c39e-c3bf-4aef-9ad0-ebc28f18818c.png)

cümlemizi girdi olarak veriyoruz ve kodlamasına karşılık gelen rakamları yukarıdaki resimde belirttik, şimdi bu her tokenize edilmiş kelimenin karşılığına gelen kodlama(embedding) bilgisayarın anlayabilmesi için vektöre çevirmemiz gerekiryor orjinal makalede 512-boyutunde kodlama vektoru kullanılmıştır her girdi kelimesi için

![resim.png](Transformer Mimarisini Anlamak_files/211a38df-e23d-4ec2-b432-e8affa0ec8f9.png)

Bu makalede biz 6 boyutlu embedding vector(kodlama vektörü) kullanıcaz ki gösterimi ve anlaşışması kolay olsun

![resim.png](Transformer Mimarisini Anlamak_files/95738fde-55d1-4809-aaac-da1811431b0b.png)

Embedding vectorü içerisinde ger kelimenin vectörünü 0 ile 1 arasında rastgele değerler ile tanımlandıgını görüyoruz ve unutmamak gerekli ki bu değerler yüksek uzayda belirli kordinatlarda yer kapsar. Ve eğitim sırasında bu yüksek uzayda kelimeler eğirim boyunca birbiine yaklaşırlar

![45.png](Transformer Mimarisini Anlamak_files/20e39ae1-4d21-47fd-a0c3-ba91e4ba997e.png)

![46.png](Transformer Mimarisini Anlamak_files/01a79b8c-1022-4408-a0db-0e63c77e8e03.png)


## 5. Adım konumsal Kodlamayı hesaplamak (Positional Embedding)

Şimdi girdilerin pozisyonal kodlamalarını bulmak gerekli, Burada 2 adet formül bulunmakta konumsal kodlamaları bulmak için her kelimenin

![resim.png](Transformer Mimarisini Anlamak_files/a78ebb26-6e82-415b-8080-c5bdd3fd1812.png)

Cümlemiz  **"When you play the game of thrones"** idi **When** kelimesini başlangıç index'i (POS) 0 ve 6 boyutlu boyuta sahip, 0 dan 5'e indexli olarak, ilk kelimemizin konumsal kodlamasını hesaplamak gerekirse girdi(input) cümlemizdeki (makaleye uygun olarak): 

- buradaki pos değeri pozisyon index değeridir
- i değeri indexi
- d değeri dimention değeri
  
2i/d konması sebebi dengeli bir frekans çıktısı sağlamak

![resim.png](Transformer Mimarisini Anlamak_files/443ea4b1-da08-4502-b5ec-78abacdce643.png)

Anlaşıldığı üzere girdi cümlemizdeki tüm kelimelerin konumsal kodlamasını hesaplıyoruz:

![resim.png](Transformer Mimarisini Anlamak_files/c639066d-6731-4cf1-974f-11015c562c85.png)


## 6. Adım Konumsal ve Kelime Gömülü Vektörlerin Birleştirilmesi

Konumsal kodlamaları hesapladıktan sonra konumsal ve kelime kodlamalarınının toplamını alıyoruz

![resim.png](Transformer Mimarisini Anlamak_files/f171cf57-e2a7-48b5-a7e5-4d0f42ebc206.png)



Özetle:

![47.png](Transformer Mimarisini Anlamak_files/92845e68-ad59-43f4-b90f-74fb989fc724.png)

## 7. Adım Multi Head Attention (Çoklu Başlı Dikkat)

Çok başlı dikkat mekanizması, birçok tek başlı dikkat mekanizmasından oluşur. Kaç tane tek başlı dikkat mekanizmasını birleştirmemiz gerektiği geliştiriciye kalmıştır. Örneğin, Meta'nın LLaMA LLM'si kodlayıcı mimarisinde 32 tek başlı dikkat mekanizması kullanmıştır. Aşağıda, tek başlı bir dikkat mekanizmasının nasıl göründüğünü gösteren şematik bir diyagram bulunmaktadır.

![resim.png](Transformer Mimarisini Anlamak_files/00769182-98ba-4c97-a8d2-7b3081682664.png)


![13_1.png](Transformer Mimarisini Anlamak_files/98b5ef47-10e7-4f0f-928c-c30e3198e708.png)

Temel olarak burada 3 Girdi bulunmakta **Q, K, V (Query, Key, Value)** bun matrislerin her biri, daha önce **kelime gömme ve konum gömme** matrislerini toplayarak hesapladığımız aynı matrisin **transpozunun** farklı bir ağırlık matrisi kümesiyle çarpılmasıyla elde edilir.

Örneğin, **Query** matrisini hesaplamak için, ağırlık matrisi kümesinin satır sayısı, **transpoz matrisinin sütun sayısıyla aynı olmalıdır**; ancak ağırlık matrisinin sütun sayısı **herhangi bir sayıda olabilir**; örneğin, ağırlık matrisimizde **4 sütun olduğunu varsayalım**. Ağırlık matrisindeki değerler **0 ile 1** arasında **rastgele** seçilir ve bu değerler, dönüştürücümüz bu kelimelerin **anlamını öğrenmeye başladığında güncellenir**.

![resim.png](Transformer Mimarisini Anlamak_files/c341ca1e-56f2-451f-943a-b3b5a58636ad.png)

Benzer şekilde key ve value matrislerini aynı şekilde hesaplayabiliriz, fakat ağırlık matris değerleri farklı olmalıdır.

![resim.png](Transformer Mimarisini Anlamak_files/c8048631-0af5-4174-8257-c788ece19309.png)

Matris çarpımları sonrası elimizde **query, key, values** değerleri olacaktır.
![resim.png](Transformer Mimarisini Anlamak_files/59161d40-ec27-483c-8c9f-9d81ce1b2e07.png)

Şimdi Elimizde Bu 3 matris mevcut single-head attention hesaplıyalım adım adım

![resim.png](Transformer Mimarisini Anlamak_files/9192ca1c-f976-42bd-b410-5ae2107b8f13.png)

Elde edilen matrisi ölçeklendirmek için, gömme vektörümüzün boyutunu, yani 6'yı yeniden kullanmalıyız.

![resim.png](Transformer Mimarisini Anlamak_files/64d39b7e-0d76-4c2b-8e1f-35f38f5d3e0e.png)

Maskeleme işleminin bir sonraki adımı **isteğe bağlıdır** ve bunu hesaplamayacağız. **Maskeleme**, modele belirli bir noktadan önce olanlara odaklanmasını ve bir cümledeki farklı kelimelerin önemini belirlerken geleceğe bakmamasını söylemek gibidir. Modelin, ileriye bakarak hile yapmadan, adım adım olayları anlamasına yardımcı olur.

Şimdi ölçeklendirilmiş sonuç matrisimize softmax işlemini uygulayacağız.

![resim.png](Transformer Mimarisini Anlamak_files/68744d36-bbc0-4a52-8800-704add54e145.png)

Tek başlı dikkat mekanizmasından elde edilen sonuç matrisini elde etmek için son çarpma adımını gerçekleştiriyoruz.

![resim.png](Transformer Mimarisini Anlamak_files/e7b43c53-eea4-4d1d-b491-768e30e2405f.png)

Daha önce belirttiğim gibi, şuana kadar **tek başlı dikkati hesapladık**, çok başlı dikkat ise **birçok tek başlı dikkatten oluşmaktadır** unutmamak gerekli. Aşağıda bunun görsel bir gösterimi yer almaktadır:

![resim.png](Transformer Mimarisini Anlamak_files/cb3b2163-8f3b-42ec-85de-e9ab1e4642b1.png)

**ama unutmamalıdır ki multi head de concat (concatenate) aşamasında tüm bu çıktılar birleştirilik en son output of multihead attention elde edilir**

![49.png](Transformer Mimarisini Anlamak_files/6a0b43a4-3544-40c6-94c2-d6f34e649b0a.png)

![48.png](Transformer Mimarisini Anlamak_files/f8a88bfe-f618-4677-b5e2-810bb5772869.png)

1. Single-head attention girdileri query, key ve value olduğunu gördük ve her biri ayrı ağırlıklara sahip ve son çıktı matrisinin nasıl elde edildiğini gördük,
2. Tek başlı dikkat mekanizmalarının tümü sonuç matrislerini ürettikten sonra, bunların hepsi birleştirilecek ve son birleştirilmiş matris,
3. Rastgele değerlerle başlatılan bir dizi ağırlık matrisiyle çarpılarak tekrar doğrusal olarak dönüştürülecektir;
4. Bu değerler daha sonra transformatör eğitime başladığında güncellenecektir.

Bizim durumumuzda **tek başlı bir dikkat mekanizmasını ele alıyoruz**, ancak çok başlı dikkat mekanizmasıyla çalışırken durum **böyle görünecektir:**

![resim.png](Transformer Mimarisini Anlamak_files/daa326ea-96cc-4ec4-9cc8-0c56f202b4ab.png)

Her iki durumda da, ister tek başlı ister çok başlı dikkat mekanizması olsun, elde edilen matrisin bir dizi ağırlık matrisiyle çarpılarak tekrar doğrusal olarak dönüştürülmesi gerekir.

![resim.png](Transformer Mimarisini Anlamak_files/4f5bf8f1-8176-451a-aea1-7dad6ad40e02.png)

Doğrusal ağırlık matrisinin sütun sayısının, daha önce hesapladığımız **(kelime gömme + konum gömme)** matrisinin sütun sayısına **eşit olduğundan emin olun**, çünkü bir sonraki adımda, elde edilen normalize edilmiş matrisi (kelime gömme + konum gömme) matrisiyle **toplayacağız.**

![resim.png](Transformer Mimarisini Anlamak_files/c12455da-870c-4014-ba57-cb714c76f2da.png)

Çok başlı dikkat mekanizması için sonuç matrisini hesapladığımıza göre, bir sonraki adımda toplama ve normalleştirme işlemlerine geçeceğiz.

## 8. Adım Toplama ve Normalize etme (Adding and normalizing)

Çok başlı dikkat mekanizması ile elde ettiğimiz sonuç matrisini, orjinal **(word embedding + positional embbeding)** matrisi ile topluyoruz öncelikle

![resim.png](Transformer Mimarisini Anlamak_files/a599b027-233c-4308-a82b-a39b50a9a4d7.png)

Elde ettiğimiz toplam martisini, ortalama ve standart sapmasını hesaplıyoruz her satır için

![resim.png](Transformer Mimarisini Anlamak_files/148e03d9-d9cf-4b95-ad16-7796a6f493bc.png)

Elde edilen ortalama ve standart sapma matrisi ile sonuç matrisini çıkartma işlemi yapılarak standard sapma değerine bölünür:

![resim.png](Transformer Mimarisini Anlamak_files/b6219ce4-f8db-43ef-b1d5-63d831e23379.png)

Küçük bir **hata payı** eklemek, paydanın sıfır olmasını engeller ve tüm terimin sonsuz olmasını önler.

## 9. Adım İleri besleme (Feed forward network)

Matrisi normalize ettikten sonra artık ileri beslemeye sokarız bu aşamada **linear layer ve bir tane ReLU aktivasyon fonksiyonu** kullanıyoruz.

![resim.png](Transformer Mimarisini Anlamak_files/2236402f-4106-4643-9f40-4e3664c0f8b5.png)

Öncelikle, son hesapladığımız matrisi, transformatör öğrenmeye başladığında güncellenecek olan **rastgele bir ağırlık matrisi kümesiyle çarparak** ve elde edilen matrisi, yine **rastgele değerler içeren bir sapma(bias)** matrisine ekleyerek doğrusal katmanı hesaplamamız gerekiyor.

![resim.png](Transformer Mimarisini Anlamak_files/28718202-2426-4321-8b86-044a28cbd537.png)

Linear katman hesaplamasından sonra **ReLU** katmanından geçirilir

![resim.png](Transformer Mimarisini Anlamak_files/50836fd0-796b-4652-9392-1e5edb1db93a.png)

## 10 Adım Tekrardan Toplama ve Normalize etme adımı

İleri besleme ağından **elde ettiğimiz matris** ile **daha önceden elde** ettiğimiz önceki toplama ve normalize etme adımında matris **toplanır** ve **satır bazlı ortalama ve standart sapma hesaplanır**

![resim.png](Transformer Mimarisini Anlamak_files/d40c1d0b-da5c-49ce-a119-077c69740b47.png)

Bu toplama ve normalleştirme adımının **çıktı matrisi**, **kod çözücü** bölümünde bulunan çok başlı dikkat mekanizmalarından birinde **sorgu ve anahtar matrisi** olarak görev yapacaktır; bunu, toplama ve normalleştirme adımlarından **kod çözücü(decoder)** bölümüne doğru izleyerek kolayca anlayabilirsiniz.

## 11. Adım Çözücü Tarafı (Decoder Part)

İyi haber şu ki, şimdiye kadar **Kodlayıcı(encoder)** kısmını hesapladık; veri setimizi kodlamaktan matrisimizi ileri beslemeli ağdan geçirmeye kadar gerçekleştirdiğimiz tüm adımlar benzersiz. Yani bunları daha önce hesaplamadık. Ancak bundan sonra, transformatörün geri kalan mimarisi olan **Kod Çözücü** kısmı, benzer türde matris çarpımlarını içerecektir.

Transformatör mimarimize bir göz atın. Şimdiye kadar neler ele aldık ve neler ele almamız gerekiyor:
![resim.png](Transformer Mimarisini Anlamak_files/00a4faf0-eb78-4b77-a27f-f447f8b3a6e9.png)

**Kod çözücünün tamamını hesaplamayacağız** çünkü büyük bir kısmı, kodlayıcıda zaten yaptığımız hesaplamalara **benzer** hesaplamalar içeriyor. Kod çözücüyü ayrıntılı olarak hesaplamak, tekrarlayan adımlar nedeniyle blogu uzatacaktır. Bunun yerine, **sadece kod çözücünün giriş ve çıkışının hesaplamalarına odaklanmamız gerekiyor.**

Eğitim sırasında, kod çözücüye iki giriş vardır. Biri kodlayıcıdan gelir; burada son toplama ve normalleştirme katmanının çıktı matrisi, kod çözücü kısmındaki ikinci çok başlı dikkat katmanı için sorgu ve anahtar görevi görür. Aşağıda bunun görselleştirilmesi yer almaktadır (** https://www.youtube.com/watch?v=gJ9kaJsE78k&t=596s **'dan):


Model:
1. Verisetindeki hedefleri görür
2. Hata yapar
3. Hata düzeltilir
4. Ağırlıklar güncellenir
5. Bu tekrar eder
Sonunda:
6. Model artık benzer girdiler için doğru çıktıyı üretebilir hale gelir.
Yani decoder input verisetindeki kelimelerden cümlelerden işlemler gerçekleştirir

![68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f323030302f302a315f5a68673936306e527179394d49462e676966.gif](Transformer Mimarisini Anlamak_files/9ebfadcf-ebb1-419e-92c7-11dcdd504e87.gif)

Değer(Value) matrisi, ilk **toplama ve normalleştirme** adımından sonra kod çözücüden gelir.

Kod çözücünün ikinci girdisi ise **tahmin edilen metindir**. Hatırlarsanız, kodlayıcıya **(encoder)** girdimiz "When you play game of thrones" idi, bu nedenle kod çözücüye girdi tahmin edilen metindir, yani bizim durumumuzda **"you win or you die".**

Ancak **tahmin edilen girdi metninin(input text)**, dönüştürücünün nerede başlayıp nerede biteceğini anlamasını sağlayan standart bir belirteç dizilimini takip etmesi gerekir. 

<start> sahte ama sabit bir başlangıç sinyali.

Düşün:
Bir yazı yazdırma robotu var.

Diyorsun ki:
- “Yazmaya başla.”
- Ama ilk harfi söylemiyorsun.
  
Robot:
- Nereden başlasın?
- Büyük harf mi?
- Küçük harf mi?
<start> = “Şimdi üretim başladı” sinyali.

![resim.png](Transformer Mimarisini Anlamak_files/d16046a1-588e-4f39-b71f-41c204788c2d.png)

Burada ve iki yeni belirteç tanıtılıyor. Dahası, **kod çözücü** her seferinde yalnızca bir belirteci girdi olarak alıyor. Bu, girdi olarak sunulacağı ve sizin de bunun için tahmin edilen metin olmanız gerektiği anlamına gelir.

![resim.png](Transformer Mimarisini Anlamak_files/1983c7df-e626-4649-acf3-10123424c477.png)

Daha önce de bildiğimiz gibi, bu gömülü vektörler rastgele değerlerle doldurulur ve bu değerler daha sonra eğitim sürecinde güncellenir.

Geri kalan blokları, kodlayıcı bölümünde daha önce hesapladığımız şekilde hesaplayın.

![resim.png](Transformer Mimarisini Anlamak_files/59312c4e-3e02-4428-845e-617006ff345f.png)

Daha ileri detaylara girmeden önce *masked multi-head attention** basit matematik ile anlaşılmalıdır

## 12. Adım Understanding Mask Multi Head Attention (Maskelenmiş çok başlı dikkat)
Masked Multi-Head Attention, decoder’da gelecekteki token’lara bakmayı engelleyen (look-ahead/causal) attention’dır.

| Normal Attention      | Masked Attention      |
| --------------------- | --------------------- |
| Herkes herkesi görür  | Gelecek gizlenir      |
| Encoder’da kullanılır | Decoder’da kullanılır |
| Çeviri input tarafı   | Üretim tarafı         |

“Bir sonraki kelimeyi tahmin ederken geleceği görmeyi engelleyen attention mekanizmasıdır.”

Transformer modellerinde, **maskeli çok başlı dikkat mekanizması**, modelin cümlenin **farklı bölümlerine odaklanmak** için kullandığı bir **spot ışığı** gibidir. Bu mekanizma özeldir çünkü modelin cümlenin ilerleyen kısımlarındaki kelimelere bakarak **hile yapmasını engeller.** Bu, modelin cümleleri adım adım anlamasına ve oluşturmasına yardımcı olur; bu da konuşma veya kelimeleri başka bir dile çevirme gibi görevlerde önemlidir.

Aşağıdaki girdi matrisine sahip olduğumuzu varsayalım; burada her satır dizideki bir pozisyonu, her sütun ise bir özelliği temsil etmektedir:

![resim.png](Transformer Mimarisini Anlamak_files/f02849ba-57b9-46d6-9532-b21ca622f66f.png)

Şimdi, iki kafalı maskeli çok kafalı dikkat bileşenlerini anlayalım:

1. **Doğrusal(Linear) Projeksiyonlar (Sorgu Query, Anahtar Key, Değer Value):** Her head için Linear projeksiyonları varsayalım: $Head 1: Wq1​,Wk1​,Wv1​$ ve $Kafa 2: Wq2​,Wk2​,Wv2​$
2. **Dikkat Puanlarını(Attention Scores) Hesapla:** Her head için, Query ve Key nokta çarpımını kullanarak attenion score hesaplayın ve gelecekteki konumlara dikkat etmeyi önlemek için maskeyi uygulayın.
3. **Softmax Uygula:** Dikkat ağırlıklarını elde etmek için softmax fonksiyonunu uygulayın.
4. **Ağırlıklı Toplam (Değer):** Her kafa için ağırlıklı toplamı elde etmek için dikkat ağırlıklarını Değer ile çarpın.
5. **Birleştir ve Doğrusal Dönüşüm:** Her iki kafadan gelen çıktıları birleştirin ve doğrusal bir dönüşüm uygulayın.

**Basitleştirilmiş bir hesaplama yapalım:**

İki koşulu varsayalım:

$Wq1​ = Wk1 ​= Wv1 ​= Wq2​ = Wk2 ​= Wv2​ = I$, birim matris.

$Q=K=V=Giriş Matrisi$

![resim.png](Transformer Mimarisini Anlamak_files/ee509de0-a686-4ee8-a449-34077c3e7992.png)

Birleştirme adımı, iki dikkat mekanizmasının çıktılarını tek bir bilgi kümesinde birleştirir. 

Bir sorun hakkında size tavsiye veren iki arkadaşınız olduğunu düşünün. 
Tavsiyelerini birleştirmek, her iki tavsiyeyi de bir araya getirerek ne önerdiklerine dair daha eksiksiz bir bakış açısı elde etmeniz anlamına gelir.
Transformer modeli bağlamında, bu adım, girdi verilerinin farklı yönlerini birden fazla perspektiften yakalamaya yardımcı olur ve modelin daha fazla işlem için kullanabileceği daha zengin bir temsile katkıda bulunur.

## 13. Adım Tahmin edilecek olan kelimeyi hesaplamak (Calculating the predicted word)

Kod çözücünün(decoder) **son** toplama ve normalleştirme bloğunun çıkış matrisi, giriş matrisiyle aynı sayıda satıra sahip olmalıdır, ancak sütun sayısı herhangi bir değer olabilir. Burada 6 ile çalışıyoruz.
![resim.png](Transformer Mimarisini Anlamak_files/f303fc7f-3bee-4469-b8e9-ed6304fe3bf5.png)

Kod çözücünün **(Decoder)** son toplama ve normalleştirme bloğunun **(add and norm)** sonuç matrisi, veri setimizdeki **(metin kümesindeki)** her benzersiz kelimenin tahmini **olasılığını** bulmak için doğrusal bir katmanla **(linear layer)** eşleşecek şekilde düzleştirilmelidir. **örneğin 4x4 matris her satır yan yana konularak birleştirilir (concatenate) edilir**

![resim.png](Transformer Mimarisini Anlamak_files/3079aa84-fff2-4cf1-9566-eab18ada5d1e.png)

Bu **düzleştirilmiş katman**, veri setimizdeki **her benzersiz kelimenin logitlerini (puanlarını) hesaplamak** için doğrusal bir katmandan **(linear layer)** geçirilecektir.

![resim.png](Transformer Mimarisini Anlamak_files/5487875f-2b2d-4161-9687-9964d2784614.png)

Logit değerlerini elde ettikten sonra, **softmax** fonksiyonunu kullanarak bunları normalleştirebilir ve **en yüksek olasılığa sahip kelimeyi bulabiliriz.**

![resim.png](Transformer Mimarisini Anlamak_files/cc97d9af-a87c-4e96-b9c8-ef631135f602.png)

Hesaplamalar sonunda tahmin edilen **kelimeyi elde ederiz.**

![resim.png](Transformer Mimarisini Anlamak_files/accc9951-f8c0-488a-a389-4d1fa975e5b2.png)

**Tahmin edilen bu kelime ("you"),** kod çözücü için giriş kelimesi olarak kabul edilecek ve bu işlem, kelime tahmin edilene kadar devam edecektir.

Önemli Noktalar

1. Yukarıdaki örnek çok basittir, çünkü Python gibi bir programlama dili kullanılarak görselleştirilebilen dönemleri veya diğer önemli parametreleri içermez.
2. Sadece eğitime kadar olan süreci göstermiştir; değerlendirme veya test, bu matris yaklaşımı kullanılarak görsel olarak görülemez.
3. Maskelenmiş çok başlı dikkat mekanizmaları, transformatörün geleceğe bakmasını engelleyerek modelinizin aşırı uyumunu önlemeye yardımcı olabilir.

# Referanslar
1. https://medium.com/@sumith.madupu123/understanding-transformer-architecture-using-simple-math-be6c2e1cdcc7
2. Attention Is All You Need - https://arxiv.org/pdf/1706.03762
3. Solving Transformer by Hand: A Step-by-Step Math Example https://github.com/FareedKhan-dev/Understanding-Transformers-Step-by-Step-math-example/blob/main/README.md
4. Visual Guide to Transformer Neural Networks https://www.youtube.com/watch?v=dichIcUZfOw
5. Transformer Architecture: The Positional Encoding https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

concatenate = birleştirmek


```python

```
