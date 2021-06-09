# Git 기초:man_student:

GitHub?   What?   Why?   How? 



**Git : 버전 관리**

**Hub : 드라이브**



## Git Bash

`ls` : 해당 폴더 안의 폴더나 파일 리스트 표시

노란색 물결(`~`) : Home 폴더 표시   →   뒤에 슬래시(`/`) 붙은 게 폴더



`Ctrl + l` 또는 `clear 입력`으로 화면을 깔끔하게 돌릴 수 있다.



***CLI는 띄어쓰기로 명령어를 구분한다!***



### 폴더 및 파일 관련 명령어

1. `ls` : 폴더 안 리스트 표시

2. `mkdir 폴더명` : 폴더 생성

3. `rm -r 폴더명` : 폴더 삭제

   :heavy_check_mark:휴지통을 거치지 않으니 주의!

4. `cd 폴더명` : 폴더 이동 / `cd ..` : 상위 폴더로 이동 / `cd ~` 또는 `cd` : 최상위 폴더로 이동

   :tipping_hand_man:**`tab` : 자동완성**

5. `touch 파일명.확장자` : 파일 생성

6. `rm 파일명.확장자` : 파일 삭제

7. Windows에서는 `start`, Mac에서는 `open`이 더블클릭(폴더 or 파일이 열림)

8. 와일드카드인 애스터리스크 `*`



### git 명령어

깃(git)은 폴더 단위로 관리하는 친구이다.

1. `git init` : 지금 있는 폴더 안에 CCTV 설치   →   파란색으로 `(master)`가 표시되고, `.git` 폴더 생성

   :star:단, git은 하위 폴더도 모두 묶기 때문에 Home 폴더는 git 하면 안 된다!

2. 이름과 이메일 생성

   `git config --global user.name 'HL'`   /   `git config --global user.email 'khl0627v@gmail.com'`

3. **`git status`** : 카메라야 지금 상태가 어때?

   :heavy_check_mark:수시로 확인해야 한다.

   만약 폴더 안의 파일이 새로 생성된 경우 `new file`, 변경된 경우 `modified`, 삭제된 경우 `deleted`로 표시된다.

   | 빨간색            | 초록색            |
   | ----------------- | ----------------- |
   | `commit`에 반영 X | `commit`에 반영 O |

4. `git add <file>` : `new file`로 추가! 이전에는 `untracked file`   →   `commit` 후에는 아무것도 뜨지 않는다.

   :soccer:**`git add .`**을 가장 많이 사용하는데, 지금 위치의 모든 내용을 Staging Area에 올린다.

5. `git commit -m '메시지'` : `-m`을 반드시 써줘야 하며, 그렇지 않은 경우 `vim`에 빠진다...

   :heavy_check_mark:메시지는 나중에 무슨 내용(업데이트)인지 알 수 있도록 하기 위함

   ​	`commit` 후 `git log`를 해보면, `HEAD` -> `master`

6.  `git log` : 사진에 대한 설명   ->   `commit 주소`, `commit 시각`, `user 정보` 등을 표시

   :heavy_check_mark:`vim`에서 빠져나오기 : `Esc`를 누르고 `:q!`를 입력한다. `git log`에서 멈췄다면 그냥 `q`를 누른다.

7. `git restore <file>` : 되살리기   ->   제거되거나 수정된 파일을 `commit`했을 때의 상태로 되돌릴 수 있다.

   `git restore --staged <file>` : 스테이지에서 내리기

8. `git checkout (commit 주소)` : **과거로의 시간여행** / `(commit 주소)`는 앞 4자리만 써도 충분하다.

   `(master)`에서 `((commit 주소))`로 바뀌고,

   :star:**과거로 돌아간다면 거기서 변경하지 말고, 복사만 하고 다시 `(master)`로 돌아와야 한다!!** 

   ​	**`git checkout master`**



## :smiley:Project pre-TODO list

1. 프로젝트 폴더(디렉토리)를 만든다.
2. `.gitignore`와 `README.md`를 생성한다.
   1. `.gitignore` 파일은 git의 파일 관리에서 무시할 내용을,
   2. `README.md`는 프로젝트의 소개 및 정리 내용을 담는다.
3. `$ git init`을 한다!
4. **주의**
   1. `.git/` 폴더와 `.gitignore` 파일, 그리고 `README.md` 파일이 같은 위치에 존재하는가!
5. 첫 번째 커밋을 한다.



![image-20210604114714995](markdown_practice.assets/image-20210604114714995.png)