import {CommerceClient as SignedClient} from "../src/lib/commerceClient";
export * from "../src/lib/commerceClient";
type Options=ConstructorParameters<typeof SignedClient>[0];
export class CommerceClient extends SignedClient {
  constructor(options:Omit<Options,"buyerId"|"accessToken">&Partial<Pick<Options,"buyerId"|"accessToken">>){
    super({buyerId:"test-buyer",accessToken:"test-token",...options});
  }
}
