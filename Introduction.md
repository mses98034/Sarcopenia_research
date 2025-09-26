請閱讀以下內容並分析整個專案，最大化符合下列需求並最大化和原先專案作整合實現最終的深度學習ASMI值預測的回歸任務

1. 資料準備：
    - 我想替換掉現有的load .pth file的方式來載入資料，主要是如果把現有資料包括image 轉成array一起打包寫入會導致空間過大，所以我希望可以只打包文字資料，然後在載入時再根據文字資料去載入對應的image。
    - 目前你可以查看/data/data.csv來了解目前準備好的資料檔，主要是裡面的IMG_PATH對應到實際DICOM影像檔案，然後其他值就如欄位所述。
    - 影像檔目前是存放在/data/Image中
    - 我再來想些修改原先的分類任務成回歸任務，所以原始專案分類任務用的category (0, 1)暫時不是我們的target y, 而是需要改成ASMI的數值。所以在Datasets準備的時候也需要把ASMI讀進來！
    - 所以最後在Datasets的撰寫也需要把分批讀進來的image tensor和text tensor合併一起丟進去訓練。
2. 模型架構：
    - 目前整體的model architecture不用大改，主要是最後的classification head需要修改成regression head,然後確保該layer的input, output size要正確。
    - 像是models/resnet的最後一個layer現在是用classifier這部份也需要作修改，改成適合我們的regression任務。
    - /module的部分主要的目的還是作feature extraction所以例如：backbone sturcture, CAM, non-local block等的這些地方不太需要修改，主要是最後的head的部分需要作修改因為他是一個classification head。 
3. 代碼修改：
    - Configuration.txt裡面是我覺得一個很重要的內容也是需要針對我們的回歸任務需要客製化修改的部分，裡面的一些hyperparameter, optimizer...是針對分類任務的，請你要修改成適合回歸任務的，包括loss function, [Network]:class, [Optimizer]等等的都需要做出相對應的修改。
    - 再來就是目前的代碼是寫死的如果有GPU就用cuda,但除了這樣的判斷邏輯外，我還需要加上mac系統的mps作為另外一個訓練device選項，也就是無論在train, test階段都可以判斷是cuda, mps如果都沒有才None用cpu.
    - 另外所有的代碼部分主要是configuration.txt和輸出的log檔裡面要針對修改後的代碼(回歸任務、device)等內容作print出來在terminal和輸出的log檔作相對應的修改。
    - 所以我覺得在/driver路徑下的ClsHelper.py, base_train_helper.py, cls_configuration.txt, test.py這些檔案都需要為了回歸任務作一定程度的修改或是創建新腳本，但若有新腳本的創建務必注意個腳本之間的import才不會有依賴缺失的問題。
4. Git ignore
    - 最主要是data/Image和data/data.csv這兩個不要上傳！
5. 模型訓練及實驗設計：
    - 目前專案的訓練方法是單純只有用一組data去作KFold作訓練和測試。我希望改成將目前的data.csv後面100筆抽出來作test dataset然後剩下的拿來做訓練可以考慮直接把剩下的用KFold去作訓練和驗證調參。
    - 目前的專案代碼是可以記錄下每次過程使用的configuration然後輸出成log檔，但我還希望可以將訓練、驗證、測試各階段的過程包括loss, 和prediction result用期刊論文常用的評估方式作繪圖，以便後續可以放在論文中作發表讓讀者可以看到更多可視化的圖表。例如使用matplotlib, seaborn等可繪出精美圖的套件。
    - 然後CAM的部分我也希望可以透過目前的代碼最後讓模型highlight出來透過我們這個image fusion text info的模型是focus在image的哪些地方讓模型的預測有更好的可解釋性。
    - 然後目前只有一個test.py的腳本，我還需要針對修改後的regression任務額外寫一個training的腳本，然後目前config和test.py都有提供使用者可以在執行腳本的時候輸入很多自訂的argument，我希望在train.py腳本也能最大化沿用這樣的方式。
6. 開發測試：
    - 在你改完代碼後需要給我指引讓我知道怎麼測試你修改的代碼是可運作的，但如果每次都要重新train一次模型然後等個好幾個小時才出現報錯，這會很沒效率，所以你要盡可能保持代碼的質量，並提供我快速檢測代碼可運行性的方式。